import sys, os, pandas as pd, streamlit as st, shap, matplotlib.pyplot as plt, plotly.graph_objects as go
from reportlab.pdfgen import canvas
from io import BytesIO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from src.ai_summary.summary_generator import predict_and_explain
from src.feature.feature_eng import feature_creation
from src.data.data_ingesion import service_col_to_binary

st.set_page_config(page_title="AI Telecom Retention Copilot",page_icon="📊",layout="wide")
st.title("📊 AI Telecom Customer Retention Copilot")
st.markdown("Predict customer churn risk and get **AI-powered retention strategies**.")
st.markdown("---")
col1,col2,col3 = st.columns(3)

with col1:
    st.subheader("Customer Profile")
    gender = st.selectbox("Gender", ["Male","Female"])
    seniorcitizen = st.selectbox("Senior Citizen", ["Yes","No"])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.slider("Tenure (Months)",0,72,12)

with col2:
    st.subheader("Telecom Services")
    phoneservice = st.selectbox("Phone Service", ["Yes","No"])
    multiplelines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    internetservice = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    onlinesecurity = st.selectbox("Online Security", ["Yes","No"])
    onlinebackup = st.selectbox("Online Backup", ["Yes","No"])
    deviceprotection = st.selectbox("Device Protection", ["Yes","No"])
    techsupport = st.selectbox("Tech Support", ["Yes","No"])

with col3:
    st.subheader("Billing")
    streamingtv = st.selectbox("Streaming TV", ["Yes","No"])
    streamingmovies = st.selectbox("Streaming Movies", ["Yes","No"])
    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperlessbilling = st.selectbox("Paperless Billing", ["Yes","No"])
    paymentmethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    monthlycharges = st.number_input("Monthly Charges",0.0,200.0,70.0)

st.markdown("---")

if st.button("🚀 Predict Churn Risk"):
    customer_data = {"gender":gender,"seniorcitizen":seniorcitizen,"partner":partner,"dependents":dependents,"tenure":tenure,"phoneservice":phoneservice,"multiplelines":multiplelines,"internetservice":internetservice,"onlinesecurity":onlinesecurity,"onlinebackup":onlinebackup,"deviceprotection":deviceprotection,"techsupport":techsupport,"streamingtv":streamingtv,"streamingmovies":streamingmovies,"contract":contract,"paperlessbilling":paperlessbilling,"paymentmethod":paymentmethod,"monthlycharges":monthlycharges}
    with st.spinner("Analyzing customer behavior..."):
        customer_df = pd.DataFrame([customer_data])
        customer_df = service_col_to_binary(customer_df)
        customer_df = feature_creation(customer_df)
        customer_data_engineered = customer_df.iloc[0].to_dict()
        result = predict_and_explain(customer_data_engineered)

    churn_prob = float(result["churn_probability"].replace("%",""))
    st.markdown("---")

    st.subheader("Customer Churn Risk Meter")
    fig = go.Figure(go.Indicator(mode="gauge+number",value=churn_prob,title={'text':"Churn Probability"},gauge={'axis':{'range':[0,100]},'steps':[{'range':[0,40],'color':"lightgreen"},{'range':[40,70],'color':"yellow"},{'range':[70,100],'color':"salmon"}]}))
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Customer Segment")
    st.info(result["risk_level"])

    st.subheader("Top Risk Factors")
    for f in result["top_risk_factors"]: st.warning(f)

    st.subheader("Model Explanation (SHAP)")
    fig,ax = plt.subplots()
    shap.plots.bar(result["shap_values"],show=False)
    st.pyplot(fig)

    st.subheader("AI Customer Insight")
    st.info(result["ai_summary"])

    st.subheader("Business Recommendations")
    recommendations = []
    if contract=="Month-to-month": recommendations.append("Offer long-term contract discount")
    if paymentmethod=="Electronic check": recommendations.append("Encourage auto-pay setup")
    if tenure<6: recommendations.append("Provide new customer welcome offer")
    if monthlycharges>80: recommendations.append("Offer plan optimization")
    for r in recommendations: st.success(r)

    # PDF generation in memory — no nested button
    st.subheader("Download Business Report")
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Telecom Customer Churn Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 770, f"Churn Probability : {result['churn_probability']}")
    c.drawString(100, 750, f"Risk Level        : {result['risk_level']}")
    c.drawString(100, 720, "Customer Details:")
    y = 700
    for k, v in customer_data.items():
        c.drawString(120, y, f"{k} : {v}")
        y -= 20
    c.drawString(100, y-10, "Top Risk Factors:")
    y -= 30
    for f in result["top_risk_factors"]:
        c.drawString(120, y, f"- {f}")
        y -= 20
    c.drawString(100, y-10, "AI Summary:")
    y -= 30
    for line in result["ai_summary"].split("\n"):
        c.drawString(120, y, line[:90])
        y -= 15
        if y < 50:
            c.showPage()
            y = 800
    c.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="customer_churn_report.pdf",
        mime="application/pdf"
    )