import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommender_system import ProfessorRecommender

# Set Page Config
st.set_page_config(
    page_title="Elmos Yar - University Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data
def load_data():
    with open(os.path.join('data', 'processed', 'normalized_comments.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

@st.cache_data
def get_recommender():
    rec = ProfessorRecommender(os.path.join('data', 'processed', 'normalized_comments.json'))
    rec.load_and_process()
    return rec

def load_clusters():
    # Load cluster plot if available
    plot_path = os.path.join('images', 'professor_clusters_plot.png')
    if os.path.exists(plot_path):
        return plot_path
    return None

import os

# --- Load Data ---
df_raw = load_data()
recommender = get_recommender()

# Flatten data for easier analysis
# Note: Recommender already processes data into a clean structure, we can reuse `recommender.df`
df = recommender.df.copy()

# Add cluster info if available
try:
    with open(os.path.join('data', 'output', 'professor_clusters.json'), 'r', encoding='utf-8') as f:
        clusters = json.load(f)
        cluster_map = {entry['professor_name']: entry['cluster'] for entry in clusters}
        df['cluster'] = df['professor_name'].map(cluster_map)
except:
    df['cluster'] = 'Unknown'

# --- Sidebar Navigation ---
st.sidebar.title("🎓 Elmos Yar")
st.sidebar.markdown("University Recommendation System")
page = st.sidebar.radio("Navigation", ["Overview", "Search Professor", "Recommender"])

# --- PAGE 1: OVERVIEW ---
if page == "Overview":
    st.title("📊 System Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Professors", len(df['professor_name'].unique()))
    with col2:
        st.metric("Total Faculties", len(df['faculty'].unique()))
    with col3:
        st.metric("Total Reviews", len(df_raw))
    with col4:
        st.metric("Avg Approval", f"{df[df['avg_eval_score'].notna()]['avg_eval_score'].mean():.1f}/10")

    # Visualizations
    st.markdown("### 📈 Data Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Faculty Distribution**")
        faculty_counts = df.groupby('faculty')['professor_name'].nunique().reset_index()
        faculty_counts.columns = ['Faculty', 'Professor Count']
        fig_fac = px.bar(faculty_counts, x='Faculty', y='Professor Count', color='Professor Count')
        st.plotly_chart(fig_fac, use_container_width=True)
        
    with col2:
        st.markdown("**Sentiment Distribution**")
        # Use raw data for full sentiment distribution
        sentiment_counts = df_raw['bert_sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig_sent = px.pie(sentiment_counts, names='Sentiment', values='Count', hole=0.4, 
                          color_discrete_map={'POSITIVE':'#00cc96', 'NEGATIVE':'#ef553b', 'NEUTRAL':'#636efa'})
        st.plotly_chart(fig_sent, use_container_width=True)
        
    st.markdown("### 🧩 Professor Clusters")
    st.info("Professors are grouped into 4 clusters based on teaching style and student feedback.")
    
    # If interactive plot is preferred, we can rebuild it here with Plotly
    # Or just show the static image generated earlier
    cluster_img = load_clusters()
    if cluster_img:
        st.image(cluster_img, caption="Professor Clusters (PCA Projection)", use_container_width=True)
    else:
        st.warning("Cluster visualization not found.")

# --- PAGE 2: SEARCH ---
elif page == "Search Professor":
    st.title("🔍 Search Professor Profile")
    
    # Search Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_faculty = st.selectbox("Filter by Faculty", ["All"] + sorted(df['faculty'].unique().tolist()))
    with col2:
        search_query = st.text_input("Search by Name", "")
        
    # Filter Logic
    filtered_df = df.copy()
    if selected_faculty != "All":
        filtered_df = filtered_df[filtered_df['faculty'] == selected_faculty]
    if search_query:
        filtered_df = filtered_df[filtered_df['professor_name'].str.contains(search_query, na=False)]
        
    # Display Results
    profs = filtered_df['professor_name'].unique()
    
    if len(profs) == 0:
        st.warning("No professors found.")
    elif len(profs) == 1 or search_query: # Show details if 1 result or searching
        selected_prof = st.selectbox("Select Professor", profs)
        
        # Get Professor Profile
        prof_data = recommender.professors[selected_prof]
        prof_raw_reviews = df_raw[df_raw['professor_name'] == selected_prof]
        
        # Header
        st.markdown(f"## 🧑‍🏫 {selected_prof}")
        st.markdown(f"**Faculty:** {prof_data['faculty']}")
        
        # Stats Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score = prof_data['avg_eval_score']
            st.metric("Avg Score", f"{score:.1f}/10" if not pd.isna(score) else "N/A")
        with col2:
            st.metric("Reviews", prof_data['review_count'])
        with col3:
            st.metric("Grading", prof_data['grading_style'].title())
        with col4:
            st.metric("Attendance", prof_data['attendance_style'].title())
            
        # Sentiment Bar
        st.markdown("### Sentiment Analysis")
        sent_counts = prof_raw_reviews['bert_sentiment'].value_counts(normalize=True)
        pos = sent_counts.get('POSITIVE', 0)
        neg = sent_counts.get('NEGATIVE', 0)
        neu = sent_counts.get('NEUTRAL', 0)
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(y=[''], x=[pos], name='Positive', orientation='h', marker_color='#00cc96'))
        fig_bar.add_trace(go.Bar(y=[''], x=[neu], name='Neutral', orientation='h', marker_color='#636efa'))
        fig_bar.add_trace(go.Bar(y=[''], x=[neg], name='Negative', orientation='h', marker_color='#ef553b'))
        fig_bar.update_layout(barmode='stack', title="Sentiment Breakdown", height=150, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Comments Section
        st.markdown("### 💬 Student Comments")
        
        tab1, tab2 = st.tabs(["Top Positive", "Top Negative"])
        
        with tab1:
            pos_reviews = prof_raw_reviews[prof_raw_reviews['bert_sentiment'] == 'POSITIVE'].sort_values('bert_score', ascending=False).head(5)
            if not pos_reviews.empty:
                for _, row in pos_reviews.iterrows():
                    st.success(row.get('description', row.get('raw_text', '')))
            else:
                st.info("No positive reviews found.")
                
        with tab2:
            neg_reviews = prof_raw_reviews[prof_raw_reviews['bert_sentiment'] == 'NEGATIVE']
            # Filter out specific neutral phrases misclassified as negative
            neg_reviews = neg_reviews[~neg_reviews['description'].astype(str).str.contains("چیزی اضافه ایی نیست", na=False)]
            neg_reviews = neg_reviews.sort_values('bert_score', ascending=False).head(5)
            
            if not neg_reviews.empty:
                for _, row in neg_reviews.iterrows():
                    st.error(row.get('description', row.get('raw_text', '')))
            else:
                st.info("No negative reviews found.")

    else:
        st.info("Select a professor from the list or refine search.")
        st.dataframe(filtered_df[['professor_name', 'faculty', 'avg_eval_score']].drop_duplicates().set_index('professor_name'))

# --- PAGE 3: RECOMMENDER ---
elif page == "Recommender":
    st.title("🤖 AI Recommender System")
    st.markdown("Find the best professor based on your course and personal priorities.")
    
    with st.form("recommender_form"):
        col1, col2 = st.columns(2)
        with col1:
            course_input = st.text_input("Course Name (e.g., 'مدار')", "")
        with col2:
            min_score = st.slider("Minimum Score", 0.0, 10.0, 5.0)
            
        col3, col4 = st.columns(2)
        with col3:
            grading_pref = st.selectbox("Grading Preference", ["Any", "lenient", "fair", "strict"])
        with col4:
            attendance_pref = st.selectbox("Attendance Preference", ["Any", "optional", "bonus", "strict"])
            
        submitted = st.form_submit_button("Find Professors")
        
    if submitted:
        priorities = {'min_score': min_score}
        if grading_pref != "Any": priorities['grading'] = grading_pref
        if attendance_pref != "Any": priorities['attendance'] = attendance_pref
        
        results, msg = recommender.recommend(course_input if course_input else None, priorities)
        
        if not results:
            st.error(msg)
        else:
            st.success(f"Found {len(results)} matches!")
            
            # Display Top Results
            for i, r in enumerate(results[:5]):
                with st.expander(f"#{i+1} {r['name']} - Score: {r['final_score']:.2f}", expanded=(i==0)):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Evaluation Score", f"{r['score']:.1f}")
                    with col2:
                        st.metric("Match Probability", f"{r['match_prob']*100:.0f}%")
                    with col3:
                        st.metric("Sentiment", f"{r['sentiment']:.2f}")
                        
                    st.write(f"**Faculty:** {r['faculty']}")
                    st.write(f"**Grading:** {r['grading']} | **Attendance:** {r['attendance']}")
