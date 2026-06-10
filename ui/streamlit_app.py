from __future__ import annotations

import streamlit as st

from app import generate_sql_pipeline
from database.excel_importer import import_excel_dataset
from database.query_results import execute_select_preview
from embeddings.faiss_store import FaissSchemaStore


def render() -> None:
    st.set_page_config(page_title="Jan Aadhaar NL2SQL", layout="wide")
    st.title("Jan Aadhaar NL2SQL")

    with st.sidebar:
        st.header("Setup")
        auto_pull    = st.checkbox("Pull missing Ollama models", value=False)
        run_profile  = st.checkbox("Profile query execution time", value=False)
        show_results = st.checkbox("Show matching entries", value=True)
        result_limit = st.number_input("Max rows to display", min_value=10, max_value=1000, value=200, step=10)

        st.divider()

        if st.button("Load dataset (Dummy_Data_Set.xlsx)"):
            with st.spinner("Loading records…"):
                try:
                    report = import_excel_dataset()
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Loaded {report.rows_loaded} citizen records.")

        if st.button("Rebuild schema index"):
            with st.spinner("Rebuilding FAISS schema index…"):
                FaissSchemaStore().build(force=True)
            st.success("Schema index rebuilt.")

    question = st.text_area(
        "Natural language question",
        value="Show all boys above 21 in Jaipur.",
        height=100,
    )

    if st.button("Generate SQL", type="primary"):
        stream_container = st.empty()
        sql_chunks: list[str] = []

        def stream_cb(chunk: str) -> None:
            sql_chunks.append(chunk)
            stream_container.code("".join(sql_chunks), language="sql")

        import time
        start_time = time.perf_counter()
        with st.spinner("Running pipeline…"):
            try:
                output = generate_sql_pipeline(
                    question,
                    ask_model_pull=auto_pull,
                    include_optimization=True,
                    run_query_for_profile=run_profile,
                    stream_callback=stream_cb,
                )
            except Exception as exc:
                st.error(str(exc))
                return
        processing_time = time.perf_counter() - start_time

        # ── Source badge ─────────────────────────────────────────────────────
        source_colors = {"cache": "🟢 Cache hit", "cache_swapped": "🟢 Smart Cache", "fast_path": "⚡ Fast Path", "llm": "🤖 LLM generated"}
        st.info(source_colors.get(output.source, output.source))

        st.subheader("Generated SQL")
        stream_container.empty()
        st.code(output.sql, language="sql")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", f"{output.confidence:.2%}")
        c2.metric("Source", output.source.upper())
        c3.metric("Retrieved columns", len(output.retrieved_columns))
        c4.metric("Time taken", f"{processing_time:.2f}s")

        if output.source == "llm" and output.retrieved_columns:
            with st.expander("Vectorised Columns Fed to LLM"):
                st.write(", ".join(output.retrieved_columns))

        if output.query_corrections:
            st.caption(f"Normalized: {output.normalized_question}")

        if output.validation_errors:
            st.error("; ".join(output.validation_errors))

        if show_results and output.sql:
            if output.is_fuzzy:
                st.subheader(f"Similarity matches for '{output.fuzzy_target}'")
                st.info("Filtered by Jaro-Winkler similarity ≥ 0.80, sorted descending.")
            else:
                st.subheader("Matching Entries")

            try:
                preview = execute_select_preview(
                    output.sql,
                    max_rows=int(result_limit),
                    fuzzy_target=output.fuzzy_target,
                    is_fuzzy=output.is_fuzzy,
                    cache_id=output.cache_id,
                )
            except Exception as exc:
                st.error(f"Could not display results: {exc}")
            else:
                if preview.rows.empty:
                    st.info("No matching entries in the loaded dataset.")
                else:
                    st.dataframe(preview.rows, use_container_width=True, hide_index=True)
                    caption = f"Showing {preview.displayed_rows} row(s)"
                    if preview.truncated:
                        caption += " — more rows exist in the database."
                    st.caption(caption)

                    st.download_button(
                        "Download as CSV",
                        data=preview.rows.to_csv(index=False).encode("utf-8"),
                        file_name="query_results.csv",
                        mime="text/csv",
                    )

                    # Auto chart: if 2 columns and one is numeric
                    import pandas as pd
                    df = preview.rows
                    if len(df.columns) == 2:
                        a, b = df.columns
                        is_num = pd.api.types.is_numeric_dtype
                        if is_num(df[b]) and not is_num(df[a]):
                            st.subheader("Data Visualization")
                            st.bar_chart(data=df, x=a, y=b)
                        elif is_num(df[a]) and not is_num(df[b]):
                            st.subheader("Data Visualization")
                            st.bar_chart(data=df, x=b, y=a)

        if output.optimization:
            with st.expander("Execution Plan"):
                st.code("\n".join(output.optimization.execution_plan))
                st.metric("Execution time", f"{output.optimization.execution_time_ms} ms")
                if output.optimization.index_recommendations:
                    st.write(output.optimization.index_recommendations)


if __name__ == "__main__":
    render()
