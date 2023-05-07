import streamlit as st
from sympy import plot_backends
from scoring_model import inference_essay, get_shap_values
import streamlit.components.v1 as components
from grammar_model import correct_essay_grammar
import shap
from IPython.display import display

# Define a function to get a motivating description for the average grade
def get_grade_description(grade):
    if grade >= 4:
        return "Great job! Keep up the excellent work! 🎉"
    elif grade >= 3:
        return "Good work! With a bit more effort, you'll do even better! 👍"
    elif grade >= 2:
        return "You're on the right track! Keep practicing and you'll see improvement! 🌟"
    else:
        return "Don't be discouraged! Keep trying and you'll make progress! 💪"

# Define a function to display the essay grades and SHAP values
def display_essay_scores(essay):
    # Grade the essay and generate SHAP values
    grades = inference_essay(essay)
    # Calculate the average grade
    avg_grade = sum(grades.values()) / len(grades)
    # Display the grades and SHAP values
    st.write(f"- Cohesion: {'✅' if grades['cohesion'] >= 4 else '🟡' if grades['cohesion'] >= 2.5 else '❌'} **{grades['cohesion']:.2f}**")
    st.write(f"- Syntax: {'✅' if grades['syntax'] >= 4 else '🟡' if grades['syntax'] >= 2.5 else '❌'} **{grades['syntax']:.2f}**")
    st.write(f"- Phraseology: {'✅' if grades['phraseology'] >= 4 else '🟡' if grades['phraseology'] >= 2.5 else '❌'} **{grades['phraseology']:.2f}**")
    st.write(f"- Vocabulary: {'✅' if grades['vocabulary'] >= 4 else '🟡' if grades['vocabulary'] >= 2.5 else '❌'} **{grades['vocabulary']:.2f}**")
    st.write(f"- Grammar: {'✅' if grades['grammar'] >= 4 else '🟡' if grades['grammar'] >= 2.5 else '❌'} **{grades['grammar']:.2f}**")
    st.write(f"- Conventions: {'✅' if grades['conventions'] >= 4 else '🟡' if grades['conventions'] >= 2.5 else '❌'} **{grades['conventions']:.2f}**")
    st.write(f"- **Average Grade:** {'✅' if avg_grade >= 4 else '🟡' if avg_grade >= 2.5 else '❌'} **{avg_grade:.2f}**")
    st.write(get_grade_description(avg_grade))
    
def display_shap_values(essay):
    st.write("### SHAP Values:")
    shap_values = get_shap_values(essay)
    st_shap(shap.plots.text(shap_values[0], display=False))

def st_shap(plot, height=None):
    print(plot)
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)
    # st.markdown(shap_html, unsafe_allow_html=True)
    

# Set up the Streamlit app
st.set_page_config(page_title="Essay Grader", page_icon="📝")
st.title("📝👨‍🏫 Essay Grader")
st.write("Enter your essay below and click 'Grade Essay' to get started:")

# Add a text input for the essay
essay = st.text_area("Enter your essay here:", height=300)

if st.button("Grade Essay"):
    # Display the essay grades and SHAP values
    st.write("### Essay Grades:")
    display_essay_scores(essay)
    
# Add a button to compute and display SHAP values
if st.button("Show SHAP Values"):
    display_shap_values(essay)

# Add a button to correct the essay's grammar
if st.button("Correct Grammar"):
    corrected_essay = correct_essay_grammar(essay)
    st.write("### Corrected Essay:")
    st.write(corrected_essay)
    st.write("#### Scores after correction:")
    # Display the corrected essay grades and SHAP values
    display_essay_scores(corrected_essay)
