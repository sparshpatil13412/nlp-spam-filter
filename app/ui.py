import streamlit as st

from predict import predict_spam

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detector")
st.write("Paste an email below and let the model judge it mercilessly.")

with st.form("spam_form"):
    email_text = st.text_area(
        "Email Content",
        height=180,
        placeholder="Enter the email text here...",
    )
    submitted = st.form_submit_button("Check Spam")

if submitted:
    if not email_text.strip():
        st.warning("Enter some text first. Even spam needs content.")
    else:
        prediction_arr, prob_arr = predict_spam(email_text)
        prediction = int(prediction_arr[0])
        spam_prob = float(prob_arr[0])

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **NOT SPAM**")

        st.metric(label="Spam Probability", value=f"{spam_prob:.1%}")

        if spam_prob > 0.7:
            st.write("Confidence: **Very High**")
        elif spam_prob > 0.4:
            st.write("Confidence: **Medium**")
        else:
            st.write("Confidence: **Low**")

st.markdown("---")
st.caption("Built with Streamlit • Powered by Machine Learning")
st.caption("UI and Model trained by Sparsh Patil")
