from google.adk.agents.llm_agent import Agent
from google.adk.agents import LoopAgent
from google.adk.tools import google_search, ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import VertexAiSearchTool
import json
from dotenv import load_dotenv
import os

load_dotenv()

DATASTORE_ID_book = os.getenv("DATASTORE_ID_book")
DATASTORE_ID_pdf = os.getenv("DATASTORE_ID_pdf")

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """
    指定されたフィールドに新しいデータを追加または上書きします。
    """
    try:
        data = json.loads(response)
        tool_context.state[field] = data
    except (json.JSONDecodeError, TypeError):
        tool_context.state[field] = response
    return {"status": "success"}

search_agent = Agent(
    model='gemini-2.5-pro',
    name='WebSearchAgent',
    instruction=
    """
    あなたは指示をもとに検索をするリサーチの専門家です。最新かつ最も正確な情報をウェブから見つけてください。
    """,
    tools=[google_search],
)

book_search_agent = Agent(
    model='gemini-2.5-pro',
    name='book_search_agent',
    description="本の内容をもとに質問に答えるエージェント",
    instruction=
    """
    あなたは所有している本の内容をもとに質問に答えるエージェントです。ユーザーからの問いかけに対し、RAGを参照し答えてください。
    ユーザーから本の指定がない場合は、様々な書籍の情報より最も適切な内容を回答してください。
    ここにある本以外の情報は使用しないでください。
    """,
    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID_book)],
)

pdf_search_agent = Agent(
    model='gemini-2.5-pro',
    name='pdf_search_agent',
    description="pdfの内容をもとに質問に答えるエージェント",
    instruction=
    """"
    あなたは所有しているpdfの内容をもとに質問に答えるエージェントです。
    ユーザーからの問いかけに対し、RAGを参照し答えてください。
    ここにあるpdf以外の情報は使用しないでください。
    """,
    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID_pdf)],
)

book_problem_agent = Agent(
    model='gemini-2.5-pro',
    name='book_problem_agent',
    description="本の内容をもとに問題を作成するエージェント",
    instruction=
    """
    あなたは所有している本の内容と改善のヒント({book_review_proposal})をもとに問題及びその解答・解説を作成するエージェントです。
    ユーザーからの問いかけに対し、RAGを参照し答えてください。
    ユーザーから本の指定がない場合は、様々な書籍の情報より作成してください。
    また、もしも問題に関して了承が得られた場合は、了承を得られた問題を再度出力してください。お礼などは不要です。
    フォーマットは以下でお願いします。 問題文、解答、解説は同時に載せてください。
    ### **問題** 
    問題内容

    <details><summary><strong>解答</strong></summary> 

    解答内容
    
    </details>

    <details><summary><strong>解説</strong></summary> 

    解説内容
    
    </details>
    """,    
    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID_book)],
    output_key="book_problem"
    )

book_problem_review_agent = Agent(
    model='gemini-2.5-pro',
    name='book_problem_review_agent',
    description="作成された問題をレビューするエージェント",
    instruction=
    """
    あなたは作成された問題と解答・解説{book_problem}に誤りがないか、
    RAGやあなた自身の考え(特に計算問題)に基づきレビューしてください。
    また、レビュー結果を具体的に述べてください。改善のヒントもあれば述べてください。
    以下のフォーマットに従っているかどうかのチェックもお願いします。
    ### **問題** 
    問題内容

    <details><summary><strong>解答</strong></summary> 

    解答内容
    
    </details>

    <details><summary><strong>解説</strong></summary> 

    解説内容
    
    </details>
    """,
    tools=[VertexAiSearchTool(data_store_id=DATASTORE_ID_book)],
    output_key="book_problem_review_result"
)

book_review_proposal_agent = Agent(
    model='gemini-2.5-pro',
    name="book_review_proposal_agent",
    instruction=
    """
    問題・解答・解説作成アドバイザーです。元の要望、作成された問題・解答・解説とレビューを基に、次の作成で試すべき具体的な改善ヒントを生成してください。
    - 作成された問題・解答・解説: {book_problem}
    - レビュー内容: {book_problem_review_result}
    """,
    output_key="book_review_proposal" # 次のループのエージェントがこれを参照する
)

book_problem_cycle_agent = LoopAgent(
    name="book_problem_cycle_agent",
    sub_agents=[
        book_problem_review_agent, # 検証
        book_review_proposal_agent,    # 次の改善ヒントを生成
        book_problem_agent,  # 改善ヒントを基に再提案
    ],
    max_iterations=4 # 無限ループを避けるための安全装置
)

curriculum_agent = Agent(
    name="curriculum_agent",
    model='gemini-2.5-pro',
    description="カリキュラムを作成するエージェント",
    instruction=
    """
    あなたは教育カリキュラムの専門家です。ユーザーの要望に基づき、最適なカリキュラムを作成してください。
    最適なカリキュラムを作成するために、まずsearch_agentを用いて必要な情報を収集してください。
    次に、その情報をもとにカリキュラムを作成してください。カリキュラムで使用する教材はbook_search_agentおよびpdf_search_agentを用いて情報収集してください。ここから収集された教材以外は使用してはいけません。
    INSTRUCTIONS:
    1. search_agentを使用してカリキュラムの立て方に関し、情報を収集してください。
    2. 収集した情報をもとに、段階的かつ効果的なカリキュラムを設計してください。カリキュラムは明確な学習目標、各ステップの内容、評価方法を含むべきです。
    3. カリキュラムで使用する教材をbook_search_agentおよびpdf_search_agentを用いて選んでください。使用しても良い教材はここにあるもののみです。ここにある教材を必ず使用してください。
    4. カリキュラムの内容は具体的かつ実践的であることを心がけてください。
    5. カリキュラムの各ステップは論理的に連続していることを確認してください。
    6. 対象のレベルやニーズに合わせたカリキュラムを作成してください。そのため、仮に対象のレベルやニーズが不明な場合は、作成前にユーザーに問い合わせてください。
    search_agent:検索を行うエージェント、ここではカリキュラムの建て方に関しての情報を収集するためのみに使用してください
    book_search_agent:本の内容をもとに質問に答えるエージェント、このカリキュラムにあった教材を提供するために使用してください
    pdf_search_agent:pdfの内容をもとに質問に答えるエージェント、このカリキュラムにあった教材を提供するために使用してください

    ** 重要 : 最後に、もう一度になりますが、カリキュラムで使用する教材はbook_search_agentおよびpdf_search_agentを用いて情報収集してください。ここから収集された教材以外は使用してはいけません。**
    """,
    tools=[AgentTool(agent=search_agent), AgentTool(agent=book_search_agent), AgentTool(agent=pdf_search_agent)],
)

config_agent = Agent(
    name="config_agent",
    model='gemini-2.5-pro',
    description="Stateに初期設定を保存するエージェント",
    instruction=
    """
    ** 重要 : このエージェントはユーザーにメッセージを出力しないでください**
    **それ以外のことは何もしないでください。**
    **あなたの役割は、append_to_stateツールを呼び出し、情報をStateに保存することだけです。**
    
    この処理はエージェントへの入力時に一度だけ実行されます。
    Stateに保存する情報は以下の通りです。これ以外は保存しないでください。
    STATE:
    book_problem : "問題はまだ作成されていません、今回の処理はスキップしてください。"
    book_problem_review_result : "レビューはまだされていません、今回の処理はスキップしてください。"
    book_review_proposal : "まずは本の内容をもとに問題・解答・解説を作成してください！"

    INSTRUCTIONS:
    1. **処理実行の判断:**
       - Stateに `book_review_proposal` が既に存在する場合、このエージェントの役割は完了しています。
       - 何も出力せず、ただちに処理を終了してください。
       - ただし、Stateの'book_review_proposal'の中身が、問題の改善のヒントに関するものである場合は実行してください。
    """,
    tools=[append_to_state],
)

root_agent = Agent(
    model='gemini-2.5-pro',
    name='root_agent',
    description="マネージャーエージェント.",
    instruction=
    """
    あなたは専門家を抱えているマネージャーです。ユーザーの問い合わせに対して専門家のエージェントにタスクをアサインしてください。
    INSTRUCTIONS:
    1.専門家へのアサインの前に、まずはconfig_agentを呼び出し、初期設定を行いましょう。
      また、初期設定をおこなっていることはユーザーに伝えないでください。
    2.** 重要 : config_agentを実行後に各専門家のエージェントにタスクをアサインしてください。config_agentの実行前に実行してはいけません。**
      各専門家のエージェントは以下の通りです。
        - book_search_agent: 本の内容をもとに質問に答えるエージェント
        - pdf_search_agent: pdfの内容をもとに質問に答えるエージェント
        - search_agent: ウェブ検索エージェント
        - book_problem_cycle_agent: 本の内容をもとに問題・解答・解説を作成し、レビューし、改善ヒントを生成するエージェント 最終的なアウトプットは `book_problem` の内容、つまり問題でお願いします。決して、問題の生成過程や、レビュー、ヒント、自己レビュー、訂正などを出力しないでください。
          book_problem_cycle_agentはユーザーからの問い合わせ一回につき一回のみの実行でお願いします。
        - curriculum_agent: カリキュラムを作成するエージェント
    3.各専門家のエージェントにタスクをアサインする際は、ユーザーの問い合わせ内容を必ず含めてください。
    4.ただいま作成していますので少々お待ちください、しばらくお待ちください、などの進捗状況を伝える言葉は絶対に言わないでください。
    5.最終的なアウトプットは体裁を整えたもののみを出力してください。決して、各エージェント間のやりとりなどを出力しないでください。
    """,
    tools=[AgentTool(agent=config_agent), 
           AgentTool(agent=book_search_agent), 
           AgentTool(agent=search_agent), 
           AgentTool(agent=pdf_search_agent), 
           AgentTool(agent=book_problem_cycle_agent),
           AgentTool(agent=curriculum_agent)],
)