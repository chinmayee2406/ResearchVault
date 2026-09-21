import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ResearchVault",
    page_icon="📚",
    layout="wide"
)


st.title("📚 ResearchVault")

st.caption(
    "Multi-Paper Research Intelligence System"
)


# --------------------------------------------------
# Upload research papers
# --------------------------------------------------

st.header("📄 Add Research Papers")

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Upload Paper"):

        with st.spinner(
            "Processing paper..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=300
                )

                response.raise_for_status()

                result = response.json()

                if "error" in result:

                    st.error(
                        result["error"]
                    )

                else:

                    st.success(
                        f"Paper '{result['paper']}' "
                        f"uploaded successfully. "
                        f"{result['chunks_added']} chunks added."
                    )

            except requests.exceptions.Timeout:

                st.error(
                    "The upload timed out. "
                    "The paper may still be processing."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the "
                    "ResearchVault API. "
                    "Make sure FastAPI is running."
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Upload failed: {error}"
                )


st.divider()


# --------------------------------------------------
# Ask questions
# --------------------------------------------------

st.header("🔎 Ask ResearchVault")


paper_options = [
    "All Papers"
]


try:

    papers_response = requests.get(
        f"{API_URL}/papers",
        timeout=10
    )

    if papers_response.status_code == 200:

        available_papers = (
            papers_response.json()
        )

        paper_options.extend(
            available_papers
        )

except requests.exceptions.RequestException:

    pass


selected_paper = st.selectbox(
    "Search in",
    paper_options
)


question = st.text_area(
    "Research Question",
    placeholder=(
        "Example: What are the main components "
        "of this paper?"
    ),
    height=120
)


if st.button(
    "🚀 Ask ResearchVault",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        if selected_paper == "All Papers":

            paper = None

        else:

            paper = selected_paper

        with st.spinner(
            "Searching papers and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "paper": paper
                    },
                    timeout=300
                )

                response.raise_for_status()

                result = response.json()

                st.divider()

                st.subheader("📝 Answer")

                st.markdown(
                    result["answer"]
                )

                st.divider()

                st.subheader("📑 Sources")

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

                            col1, col2, col3 = (
                                st.columns(3)
                            )

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
                    "The request timed out."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the "
                    "ResearchVault API."
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"API request failed: {error}"
                )