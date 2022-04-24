import streamlit as st
from function import *
import pickle
loaded_model = pickle.load(open("model.pkl", 'rb'))
loaded_vect = pickle.load(open("transformer.pkl", 'rb'))
def main():
    st.title("Food review Prediction ")
    user_review= str(st.text_area("Enter your review"))
    if st.button("predict"):
        x=raw_test(user_review, loaded_model, loaded_vect)
        st.success(x)
if __name__ == '__main__':
    main()
