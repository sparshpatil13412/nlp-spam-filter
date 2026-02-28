from predict import predict_spam
import streamlit as st

st.set_page_config(
    page_title="Spam Detector",
    page_icon= "📧",
    layout="centered"
)

st.title("📧 Email Spam Detector")
st.write("Paste an email below and let the model judge it mercilessly.")

email_text = st.text_area(
    "Email Content",
    height=180,
    placeholder="Enter the email text here..."
)

if st.button("Check Spam"):
    if email_text.strip() == "":
        st.warning("Enter some text first. Even spam needs content.")
    else:
        prediction, prob = predict_spam(email_text)

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **NOT SPAM**")

        st.metric(
            label="Spam Probability",
            value = f"{prob.item():.2f}"
        )

        if prob > 0.7:
            st.write("Confidence: **Very High**")
        elif prob > 0.4:
            st.write("Confidence: **Medium**")
        else:
            st.write("Confidence: **Low**")

st.markdown("---")
st.caption("Built with Streamlit • Powered by Machine Learning")
st.caption(" UI and Model trained by Sparsh Patil")