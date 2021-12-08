import streamlit as st

import requests
import json

from st_gen_page import *
from st_clf_page import *

############### Page Initialization
st.set_page_config(page_title="Defeasible Inference Demo", layout='wide')

desc = "# This is Defeasible Inference Demo Page."
# st.title('Defeasible Inference Demo')

st.write(desc)

############### Making Tabbed Subpage
# https://github.com/streamlit/streamlit/issues/233

st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
    unsafe_allow_html=True,
)

query_params = st.experimental_get_query_params()
# tabs = ["Home", "About", "Contact"]
tabs = ["Generation Model", "Classifier Model"]
if "tab" in query_params:
    active_tab = query_params["tab"][0]
else:
    active_tab = "Generation Model"

if active_tab not in tabs:
    st.experimental_set_query_params(tab="Generation Model")
    active_tab = "Generation Model"

li_items = "".join(
    f"""
    <li class="nav-item">
        <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
    </li>
    """
    for t in tabs
)

tabs_html = f"""
    <ul class="nav nav-tabs">
    {li_items}
    </ul>
"""

st.markdown(tabs_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

############################################################

#Make Page Based on Active Tab
if active_tab == "Generation Model":
    st.markdown("### Model Structure")
    st.image("./imgs/defeasible_clf.png", caption="Defeasible Inference Classifier Model", width=500)

    submitted_gen, model_option, inputs = make_gen_page()
    if submitted_gen:
        st.markdown("## Pressed GEN!")

    # Define Submit Button Behavior
    if submitted_gen:
        # Send Request to Server
        if model_option=="delta-SNLI Trained":
            port=1999
        else:
            port=1999
        url = f'http://127.0.0.1:{port}/predict_gen'
        with st.spinner('Generating Weakener...'):
            x = requests.post(url, json = inputs)

        # Parse JSON and unpack
        got_dict=json.loads(x.text)
        val=got_dict["test_val"]
        additional_info=got_dict["additional_info"]

        # Present output
        output = val
        st.markdown(f"<div style='height: 100px; background-color: #F0F2F6; margin-bottom: 20px; overflow: scroll;'><p style='padding: 10px; overflow: scroll;'>{output}</p></div>", unsafe_allow_html=True)
        
        with st.expander("See Inference Debug Info"):
            st.markdown("* Generation Decoding Parameters: "+str(additional_info["decode_params"]))
            st.markdown("* Inference Time: "+"{:.3f} seconds".format(additional_info["inf_time"]))
            st.markdown("* Accelerator: "+additional_info["device"])

elif active_tab == "Classifier Model":
    submitted_clf, model_option, inputs = make_clf_page()
    if submitted_clf:
        if model_option=="delta-SNLI Trained":
            port=2000
        else:
            port=2000

        url = f'http://127.0.0.1:{port}/predict_clf'
        with st.spinner('Predicting Update...'):
            x = requests.post(url, json = inputs)

        # Parse JSON and unpack
        got_dict=json.loads(x.text)
        val=got_dict["predicted"]
        probs=got_dict["probs"]
        additional_info=got_dict["additional_info"]

        # Present output
        output = ""
        if val==0:
            output += "Predicted: Strengthener"
        else:
            output += "Predicted: Weakener"

        output+="     Probs: "+str(probs)
        st.markdown(f"<div style='height: 100px; background-color: #F0F2F6; margin-bottom: 20px; overflow: scroll;'><p style='padding: 10px; overflow: scroll;'>{output}</p></div>", unsafe_allow_html=True)
        
        with st.expander("See Inference Debug Info"):
            st.markdown("* Inference Time: "+"{:.3f} seconds".format(additional_info["inf_time"]))
            st.markdown("* Accelerator: "+additional_info["device"])

else:
    st.error("Something has gone terribly wrong.")


st.markdown("만든 사람: 성균관대학교 인공지능학과 ING Lab 석사과정 [송영록](https://github.com/id4thomas)")




