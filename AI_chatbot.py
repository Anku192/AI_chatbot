import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


if 'faqs' not in st.session_state:
    st.session_state.faqs = [
        {"question": "Hi", "answer": "Hello good evening sir/ma'am"},
        {"question": "Hello", "answer": "Hello good evening sir/ma'am"},
        {"question": "how are you", "answer": "i am fine today"},
        {"question": "How can I place an order?", "answer": "Browse the products, add them to your cart, and proceed to checkout to place an order"},
        {"question": "Hi", "answer": "Hello good evening sir/ma'am"},
    {"question": "Hello", "answer": "Hello good evening sir/ma'am"},
    {"question": "how are you", "answer": "i am fine today"},
    {"question": "How can I place an order?", "answer": "Browse the products, add them to your cart, and proceed to checkout to place an order"},
    {"question": "What payment methods do you accept?", "answer": "We accept Credit/Debit Cards, Net Banking, UPI, Wallets, and Cash on Delivery (COD)."},
    {"question": "Can I cancel my order after placing it?", "answer": "You can cancel your order from the 'My Orders' section before it is shipped."},
    {"question": "How do I apply a discount coupon?", "answer": "You can apply your coupon code during checkout in the Apply Coupon  field."},
    {"question": "What are your delivery charges?", "answer": "Delivery is free for orders above ₹500. For orders below that, a nominal fee of ₹50 applies."},
    {"question": "How long does delivery take?", "answer": "Delivery usually takes 3-5 business days, depending on your location."},
    {"question": "What is your return policy?", "answer": "You can return products within 7 days of delivery for a full refund, provided they are unused and in original packaging."},
    {"question": "How will I get my refund?", "answer": "Refunds will be credited to your original payment method within 5-7 business days after we receive the returned item."},
    {"question": "Can I exchange a product?", "answer": "Yes, you can request an exchange for size or color variations via the 'My Orders' section."},
    {"question": "How do I know if a product is in stock?", "answer": "Product availability is mentioned on the product page. If out of stock, you'll see an option to get notified when it’s back."},
    {"question": "Do you provide product warranties?", "answer": "Warranty details are mentioned in the product description, where applicable."},
    {"question": "How can I contact customer service?", "answer": "You can reach us via live chat, email at support@example.com, or call us at +91-9876543210."},
    {"question": "Do you have a mobile app?", "answer": "Yes, our mobile app is available for both Android and iOS devices."},
    {"question": "Is my payment information secure?", "answer": "Absolutely. We use SSL encryption and comply with PCI DSS standards to protect your payment data."},
    {"question": "Will my personal information be shared?", "answer": "We do not share your personal information with third parties without your consent. Please refer to our Privacy Policy."},
    {"question": "Do you offer gift cards?", "answer": "Yes, you can purchase digital gift cards from our Gift Card section."},
    {"question": "Can I change my delivery address after placing an order?", "answer": "You can change your address before the order is shipped by contacting our support team."},
    {"question": "thank you", "answer": "you are welcome, it's a pleasure assisting you"},
    {"question": "i have some doubts", "answer": "sure please ask i am happy to help"},
    {"question": "what is your business about", "answer": "we are an online shopping service with a wide range of products"},
    {"question": "I forgot my password. How can I recover it?", "answer": "please Click on forgot password and follow the instructions to reset your password"},
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 6 PM, Monday to Friday."},
    {"question": "How can I reset my password?", "answer": "You can reset your password by clicking 'Forgot Password' on the login page."},
    {"question": "Where is your company located?", "answer": "We are located in Mumbai, India."}
        
        
    ]


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def update_embeddings():
    faq_questions = [faq["question"] for faq in st.session_state.faqs]
    st.session_state.faq_embeddings = model.encode(faq_questions)


if 'faq_embeddings' not in st.session_state:
    update_embeddings()

def get_best_match(user_query):
    user_embedding = model.encode([user_query])
    similarities = cosine_similarity(user_embedding, st.session_state.faq_embeddings)
    best_match_idx = np.argmax(similarities)
    similarity_score = similarities[0][best_match_idx]
    
    if similarity_score > 0.6: 
        return st.session_state.faqs[best_match_idx]["answer"], best_match_idx, similarity_score
    else:
        return "Sorry, I don't have an answer for that yet.", -1, similarity_score


st.title("🤖 AI Chatbot Trainer Interface")


tab1, tab2, tab3 = st.tabs(["Chat with Bot", "Train the Bot", "FAQ Management"])

with tab1:
    st.header("Chat with the Bot")
    user_input = st.text_input("Ask me something...", key="chat_input")
    
    if user_input:
        response, match_idx, similarity = get_best_match(user_input)
        st.write(f"**Bot:** {response}")
        
        
        st.write(f"**Matching Info:**")
        col1, col2 = st.columns(2)
        col1.metric("Similarity Score", f"{similarity:.2f}")
        
        if match_idx != -1:
            matched_question = st.session_state.faqs[match_idx]["question"]
            col2.metric("Matched Question", matched_question)
        else:
            col2.metric("Matched Question", "No close match")

with tab2:
    st.header("Train the Bot")
    st.write("Here you can improve the bot's responses by adding new questions or updating existing answers.")
    
    
    if 'chat_input' in st.session_state and st.session_state.chat_input:
        st.write(f"**Last User Query:** {st.session_state.chat_input}")
        last_response, last_match_idx, _ = get_best_match(st.session_state.chat_input)
        st.write(f"**Bot's Response:** {last_response}")
    
   
    with st.form("train_form"):
        if 'chat_input' in st.session_state and st.session_state.chat_input:
            new_question = st.text_input("Question", value=st.session_state.chat_input)
        else:
            new_question = st.text_input("Question")
        
        new_answer = st.text_area("Answer")
        
     
        if last_match_idx != -1 and 'chat_input' in st.session_state:
            update_existing = st.checkbox("Update existing FAQ instead of creating new one")
        else:
            update_existing = False
        
        submitted = st.form_submit_button("Save Training")
        
        if submitted and new_question and new_answer:
            if update_existing and last_match_idx != -1:
              
                st.session_state.faqs[last_match_idx]["answer"] = new_answer
                st.success(f"Updated answer for: '{st.session_state.faqs[last_match_idx]['question']}'")
            else:
              
                st.session_state.faqs.append({"question": new_question, "answer": new_answer})
                st.success("Added new Q&A pair to the knowledge base!")
            
            update_embeddings()

with tab3:
    st.header("FAQ Management")
    st.write("View, edit, and manage all the questions and answers in the knowledge base.")
    
    df = pd.DataFrame(st.session_state.faqs)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="faq_editor"
    )
    
    if st.button("Save All Changes", key="save_faqs"):
        st.session_state.faqs = edited_df.to_dict('records')
        update_embeddings()
        st.success("FAQ knowledge base updated!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="Export FAQs as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='faq_knowledge_base.csv',
            mime='text/csv'
        )
    
    with col2:
        uploaded_file = st.file_uploader("Import FAQs from CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                new_faqs = pd.read_csv(uploaded_file).to_dict('records')
                st.session_state.faqs = new_faqs
                update_embeddings()
                st.success("FAQ knowledge base imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing FAQs: {str(e)}")

st.sidebar.header("Analytics")
st.sidebar.metric("Total Q&A Pairs", len(st.session_state.faqs))
st.sidebar.write("**Recent Additions:**")
for faq in st.session_state.faqs[-5:][::-1]:  
    st.sidebar.write(f"Q: {faq['question']}")
    st.sidebar.write(f"A: {faq['answer']}")
    st.sidebar.write("---")