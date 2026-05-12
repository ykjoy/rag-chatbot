"""
==============================================================================
사업보고서 RAG 챗봇 (LlamaIndex + Gemini API + Supabase pgvector)
==============================================================================
- PDF 파일 업로드 (DART 사업보고서) 또는 HTML URL 입력 (SEC 10-K 등)
- LlamaIndex가 자동으로 청킹·임베딩 → Supabase pgvector 저장
- Gemini가 출처 페이지와 함께 답변 생성
- 대화 이력은 Supabase chat_history 테이블에 저장
==============================================================================
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from supabase import create_client, Client
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.readers.web import SimpleWebPageReader

# --------------------------------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="사업보고서 RAG 챗봇",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------------------------------
# 2. 비밀 키(Secrets) 불러오기
# --------------------------------------------------------------------------
# Streamlit Cloud의 Secrets(또는 로컬의 .streamlit/secrets.toml)에서 키를 읽어옵니다.
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Supabase Direct Connection String (벡터 저장용)
# Supabase 대시보드 → Project Settings → Database → Connection string에서 확인 가능
# 형식: postgresql://postgres:[YOUR-PASSWORD]@db.xxx.supabase.co:5432/postgres
SUPABASE_DB_CONNECTION = st.secrets["SUPABASE_DB_CONNECTION"]


# --------------------------------------------------------------------------
# 3. Supabase 및 LlamaIndex 초기화
# --------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    """Supabase 클라이언트 초기화 (캐싱하여 속도 향상)"""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def init_llama_index():
    """LlamaIndex 전역 설정 — LLM과 임베딩 모델을 Gemini로 지정"""
    # LLM: 답변 생성용 (gemini-2.5-flash — 무료 티어, 빠름)
    Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1,  # 환각 최소화
    )
    # 임베딩 모델: 텍스트를 768차원 벡터로 변환
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        api_key=GEMINI_API_KEY,
    )
    # 청크 크기 (한 조각의 크기)
    Settings.chunk_size = 500
    Settings.chunk_overlap = 50


@st.cache_resource
def get_vector_store(company_name: str):
    """Supabase pgvector에 연결된 LlamaIndex 벡터 스토어 반환"""
    return SupabaseVectorStore(
        postgres_connection_string=SUPABASE_DB_CONNECTION,
        collection_name=company_name.replace(" ", "_").lower(),
        dimension=768,
    )


supabase = init_supabase()
init_llama_index()


# --------------------------------------------------------------------------
# 4. 화면 UI 구성 (3개의 탭)
# --------------------------------------------------------------------------
st.title("📊 사업보고서 RAG 챗봇")
st.info(
    "💡 안내: PDF 사업보고서를 업로드하거나 10-K URL을 입력하면, "
    "AI가 내용을 학습하고 자연어로 질문에 답해드립니다."
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "index" not in st.session_state:
    st.session_state.index = None

tab1, tab2, tab3 = st.tabs(["📤 업로드", "💬 챗봇", "📜 채팅 기록"])

# ==========================================================================
# 탭 1: 사업보고서 업로드 (PDF 또는 HTML URL)
# ==========================================================================
with tab1:
    st.subheader("사업보고서 인덱싱")

    company_name = st.text_input(
        "🏢 회사명 입력",
        placeholder="예: 삼성전자, Apple, 네이버",
        help="이 보고서가 어느 회사의 것인지 표시하기 위한 라벨입니다.",
    )

    # 입력 방식 선택: PDF 파일 또는 HTML URL
    input_method = st.radio(
        "📥 데이터 입력 방식",
        ["📄 PDF 파일 업로드 (DART 사업보고서)", "🌐 HTML URL 입력 (SEC 10-K 등)"],
        horizontal=True,
    )

    # ---------- 방식 1: PDF 업로드 ----------
    if input_method.startswith("📄"):
        uploaded_file = st.file_uploader(
            "PDF 파일 선택",
            type=["pdf"],
            help="DART(전자공시시스템)에서 다운로드한 사업보고서 PDF를 업로드하세요.",
        )

        if uploaded_file is not None and company_name:
            if st.button("🚀 PDF 인덱싱 시작", type="primary"):
                with st.spinner("PDF 읽는 중... (약 1-2분 소요)"):
                    try:
                        # PDF를 임시 폴더에 저장 (LlamaIndex가 폴더 단위로 읽기 때문)
                        with tempfile.TemporaryDirectory() as temp_dir:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            # 1. PDF 읽기
                            documents = SimpleDirectoryReader(
                                input_dir=temp_dir
                            ).load_data()

                            # 2. 메타데이터에 회사명 추가 (검색 결과 추적용)
                            for doc in documents:
                                doc.metadata["company"] = company_name
                                doc.metadata["source_type"] = "PDF"

                            # 3. 벡터 스토어 + 인덱스 생성
                            vector_store = get_vector_store(company_name)
                            storage_context = StorageContext.from_defaults(
                                vector_store=vector_store
                            )
                            index = VectorStoreIndex.from_documents(
                                documents,
                                storage_context=storage_context,
                                show_progress=True,
                            )

                            st.session_state.index = index
                            st.session_state.current_company = company_name

                        st.success(
                            f"✅ '{company_name}' PDF 인덱싱 완료! "
                            f"({len(documents)} 페이지 처리)"
                        )
                        st.info("💬 챗봇 탭으로 이동해서 질문해보세요.")

                    except Exception as e:
                        st.error(f"오류 발생: {e}")

    # ---------- 방식 2: HTML URL 입력 ----------
    else:
        url_input = st.text_input(
            "🔗 보고서 URL 입력",
            placeholder="예: https://www.sec.gov/Archives/edgar/data/.../aapl-20240928.htm",
            help="SEC EDGAR에서 10-K 보고서의 HTML 페이지 URL을 복사해서 붙여넣으세요.",
        )

        if url_input and company_name:
            if st.button("🚀 URL 인덱싱 시작", type="primary"):
                with st.spinner("웹페이지 읽는 중... (약 1-2분 소요)"):
                    try:
                        # 1. HTML 웹페이지 읽기
                        documents = SimpleWebPageReader(html_to_text=True).load_data(
                            [url_input]
                        )

                        # 2. 메타데이터 추가
                        for doc in documents:
                            doc.metadata["company"] = company_name
                            doc.metadata["source_type"] = "HTML"
                            doc.metadata["url"] = url_input

                        # 3. 벡터 스토어 + 인덱스 생성
                        vector_store = get_vector_store(company_name)
                        storage_context = StorageContext.from_defaults(
                            vector_store=vector_store
                        )
                        index = VectorStoreIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            show_progress=True,
                        )

                        st.session_state.index = index
                        st.session_state.current_company = company_name

                        st.success(f"✅ '{company_name}' HTML 인덱싱 완료!")
                        st.info("💬 챗봇 탭으로 이동해서 질문해보세요.")

                    except Exception as e:
                        st.error(f"오류 발생: {e}")


# ==========================================================================
# 탭 2: 챗봇 (RAG로 질문 답변)
# ==========================================================================
with tab2:
    st.subheader("💬 사업보고서에 질문하기")

    if st.session_state.current_company:
        st.caption(f"📁 분석 대상: **{st.session_state.current_company}**")
    else:
        st.warning("⚠ 먼저 '업로드' 탭에서 사업보고서를 인덱싱해주세요.")

    # 이전 대화 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 참고 출처"):
                    for src in msg["sources"]:
                        st.caption(src)

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요 (예: 작년 매출은?)"):
        if not st.session_state.index:
            st.error("먼저 사업보고서를 업로드해주세요.")
        else:
            # 사용자 메시지 표시
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 답변 생성
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        # RAG 쿼리 실행
                        query_engine = st.session_state.index.as_query_engine(
                            similarity_top_k=5,  # 관련 청크 5개 검색
                        )
                        response = query_engine.query(prompt)

                        answer = str(response)
                        st.markdown(answer)

                        # 출처 표시
                        sources = []
                        for node in response.source_nodes:
                            page = node.metadata.get("page_label", "?")
                            src_type = node.metadata.get("source_type", "")
                            sources.append(
                                f"{src_type} 페이지 {page}: {node.text[:100]}..."
                            )

                        if sources:
                            with st.expander("📄 참고 출처"):
                                for src in sources:
                                    st.caption(src)

                        # 메시지에 저장
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )

                        # Supabase chat_history에 저장
                        try:
                            supabase.table("chat_history").insert(
                                {
                                    "question": prompt,
                                    "answer": answer,
                                    "sources": sources,
                                    "company_name": st.session_state.current_company,
                                }
                            ).execute()
                        except Exception as db_e:
                            st.toast(f"DB 저장 실패: {db_e}")

                    except Exception as e:
                        st.error(f"답변 생성 오류: {e}")


# ==========================================================================
# 탭 3: 채팅 기록 (Supabase에 저장된 과거 대화)
# ==========================================================================
with tab3:
    st.subheader("📜 채팅 기록")

    try:
        response = (
            supabase.table("chat_history")
            .select("*")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        if response.data:
            df = pd.DataFrame(response.data)
            df = df[["created_at", "company_name", "question", "answer"]]
            df.columns = ["시간", "회사", "질문", "답변"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            # CSV 다운로드
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 CSV로 다운로드",
                csv,
                "chat_history.csv",
                "text/csv",
            )
        else:
            st.info("아직 저장된 대화가 없습니다. 챗봇 탭에서 질문해보세요!")

    except Exception as e:
        st.error(f"채팅 기록 불러오기 실패: {e}")
