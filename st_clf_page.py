import streamlit as st
# from defeasible_gen_model import *
import requests
import json


def make_clf_page():
    st.write("Demo for Defeasible Classifier Model!")


    main_column, model_column = st.columns([4, 1])

    # IN :1
    with model_column:
        st.markdown(" ")
        # st.markdown(" ")
        st.markdown("#### Select Model")
        model_option = st.selectbox(label="", options=["delta-SNLI Trained", "delta-ATOMIC Trained"])

    # IN 4:
    with main_column:
        with st.form("gen_form"):
            st.markdown("### Select From Examples")

            # Show Examples based on selected
            if model_option=="delta-SNLI Trained":
                st.markdown("Example 1")
                st.markdown("* Premise: [MALE] has a very important math test next week.")
                st.markdown("* Hyothesis: [MALE] get a good grade on the test.")
                st.markdown("* Update: The math test is very difficult")

                st.markdown("Example 2")
                st.markdown("* Premise: [FEMALE] decided she was finally ready to get a pet.")
                st.markdown("* Hyothesis: [FEMALE] go to the pet store and buy a pet.")
                st.markdown("* Update: The pet store is closed.")
            else:
                st.markdown("Example 1")
                st.markdown("* Premise: [MALE] has a very important math test next week. ATOMIC")
                st.markdown("* Hyothesis: [MALE] get a good grade on the test.")
                st.markdown("* Update: The math test is very difficult")

                st.markdown("Example 2")
                st.markdown("* Premise: [FEMALE] decided she was finally ready to get a pet.")
                st.markdown("* Hyothesis: [FEMALE] go to the pet store and buy a pet.")
                st.markdown("* Update: The pet store is closed.")


            # Radio Button of Examples
            example_select = st.radio("Select from above examples", options=["Example 1","Example 2","Test your own"])


            # premise = st.selectbox(label="", options=[
            #         "[MALE] has a very important math test next week.This is test",
            #         "[FEMALE] decided she was finally ready to get a pet.",
            #         "Enter Input.."])
            input_placeholder = st.empty()
            premise_input=""
            hyp_input=""
            update_input=""

            if example_select=="Test your own":
                st.markdown("#### Example your own input")
                premise_input = st.text_input("Enter Premise (Seed) here.", help = "Enter Premise (Seed) here.")
                hyp_input = st.text_input("Enter Hypothesis (Goal) here.", help = "Enter Hypothesis (Goal) here.")
                update_input = st.text_input("Enter Update here.", help = "Enter Update here.")

            submitted_clf = st.form_submit_button("실행_CLF")

        inputs={
            'example_select': example_select,
            'premise_input': premise_input,
            'hyp_input': hyp_input,
            'update_input': update_input,
        }

    return submitted_clf, model_option, inputs