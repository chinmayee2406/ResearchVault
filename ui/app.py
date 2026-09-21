import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask"


st.set_page_config(
    page_title="ResearchVault",
    page_icon="📚",
    layout="wide"
)


st.title("📚 ResearchVault")
st.caption(
    "Multi-Paper Research Intelligence System"
)

st.markdown(
    """
    Ask questions across research papers and receive
    evidence-backed answers with page-level sources.
    """
)


st.divider()


question = st.text_area(
    "🔎 Research Question",
    placeholder=(
        "Example: What are the main components "
        "of a RAG system?"
    ),
    height=120
)


ask_button = st.button(
    "🚀 Ask ResearchVault",
    type="primary"
)


if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        with st.spinner(
            "Searching papers and generating answer..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=300
                )

                response.raise_for_status()

                result = response.json()

                st.divider()

                st.header("📝 Answer")

                st.markdown(
                    result["answer"]
                )

                st.divider()

                st.header("📑 Sources")

                sources = result.get(
                    "sources",
                    []
                )

                if not sources:

                    st.info(
                        "No sources were returned."
                    )

                else:

                    for index, source in enumerate(
                        sources,
                        start=1
                    ):

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"**Source {index}**"
                            )

                            col1, col2, col3 = st.columns(3)

                            with col1:

                                st.markdown(
                                    "**Paper**"
                                )

                                st.write(
                                    source["paper"]
                                )

                            with col2:

                                st.markdown(
                                    "**Page**"
                                )

                                st.write(
                                    source["page"]
                                )

                            with col3:

                                st.markdown(
                                    "**Chunk**"
                                )

                                st.write(
                                    source["chunk_id"]
                                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "The local RAG pipeline may still "
                    "be processing."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the "
                    "ResearchVault API. Make sure "
                    "FastAPI is running on port 8000."
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"API request failed: {error}"
                )