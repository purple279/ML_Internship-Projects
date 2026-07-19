from unicodedata import category, numeric
from narwhals import corr
from scipy import cluster
import streamlit as st
from streamlit_option_menu import option_menu
import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "final_dataset.csv")
original_df = pd.read_csv(BASE_DIR / "original_dataset.csv")
processed_df = pd.read_csv(BASE_DIR / "learner_dataset.csv")

st.set_page_config(
    page_title = "EduPro Platform",
    page_icon = "🎓",
    layout = "wide"
)

def home_page():

    st.title("EduPro Student Segmentation & Personalized Course Recommendation System")

    st.markdown("""
    Welcome to the EduPro Dashboard.

    This dashboard helps analyze learner behavior, perform student segmentation,
    and recommend personalized learning paths.
    """)

    # ===========================
    # KPI Cards
    # ===========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👨‍🎓 Learners", len(processed_df))

    with col2:
        st.metric(
            "📚 Avg Courses",
            round(processed_df["Total_courses_enrolled"].mean(), 2)
        )

    with col3:
        st.metric(
            "⭐ Avg Rating",
            round(original_df["CourseRating"].mean(), 2)
        )

    with col4:
        st.metric(
            "💰 Revenue",
            f"₹{original_df['Amount'].sum():,.0f}"
        )

    st.divider()

    # ===========================
    # Course Category Distribution
    # ===========================

    category_df = (
        original_df["CourseCategory"]
        .value_counts()
        .reset_index()
    )

    category_df.columns = ["Category", "Count"]

    fig = px.pie(
        category_df,
        values="Count",
        names="Category",
        title="Course Category Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # Age Group Distribution
    # ===========================

    age_df = (
        processed_df["Age_group"]
        .value_counts()
        .reset_index()
    )

    age_df.columns = ["Age Group", "Count"]

    fig = px.bar(
        age_df,
        x="Age Group",
        y="Count",
        title="Learner Age Group Distribution",
        text="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # Monthly Enrollment Trend
    # ===========================

    monthly_df = (
        processed_df.groupby("Month")
        .size()
        .reset_index(name="Enrollments")
    )

    fig = px.line(
        monthly_df,
        x="Month",
        y="Enrollments",
        markers=True,
        title="Monthly Enrollment Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # Course Level Distribution
    # ===========================

    level_df = (
        original_df["CourseLevel"]
        .value_counts()
        .reset_index()
    )

    level_df.columns = ["Course Level", "Count"]

    fig = px.pie(
        level_df,
        values="Count",
        names="Course Level",
        hole=0.4,
        title="Course Level Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # About Project
    # ===========================

    st.subheader("About Project")

    st.write("""
This project segments learners using K-Means clustering and recommends personalized courses based on learner behaviour, preferences, and engagement patterns.

The dashboard provides interactive visualizations, learner profile exploration, cluster analysis, and personalized course recommendations to help EduPro improve learner engagement and decision-making.
""")
       
def dataset_page():

    st.title("📂 Dataset Explorer")
    st.write("Explore the original and processed datasets used in this project.")

    st.subheader('Dataset Overview')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric('Rows', original_df.shape[0])

    with col2:
        st.metric('Columns', original_df.shape[1])

    with col3:
        st.metric('Missing values', original_df.isnull().sum().sum())

    with col4:
        st.metric('Duplicate Rows', original_df.duplicated().sum())

    dataset_choice = st.selectbox(
        'select dataset to explore',
        ['Original Dataset', 'Learner Dataset', 'Final Dataset']
    )               

    if dataset_choice == 'Original Dataset':
        st.dataframe(original_df)

    elif dataset_choice == 'Learner Dataset':
        st.dataframe(processed_df)

    else:
        st.dataframe(df)

    st.subheader('Column Information')

    column_info = pd.DataFrame({
        'Column': original_df.columns,
        'Data Type': original_df.dtypes.astype(str)
    })

    st.dataframe(column_info)

    st.subheader("Summary Statistics")
    st.dataframe(original_df.describe())

    st.subheader("Missing Values")

    missing = pd.DataFrame({
        "Column": original_df.columns,
        "Missing Values": original_df.isnull().sum()
    })

    st.dataframe(missing)

    st.download_button(
        label = "📥 Download Processed Dataset",
        data=processed_df.to_csv(index=False),
        file_name="learner_dataset.csv",
        mime="text/csv"
    )

def eda_page():

    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Explore learner behaviour and course statistics.")

    # ---------------- Row 1 ---------------- #

    col1, col2 = st.columns(2)

    with col1:

        gender = {
            "Male": original_df["Gender"].value_counts().get("Male", 0),
            "Female": original_df["Gender"].value_counts().get("Female", 0)
        }

        gender_df = pd.DataFrame(
            gender.items(),
            columns=["Gender", "Count"]
        )

        fig = px.pie(
            gender_df,
            names="Gender",
            values="Count",
            title="Gender Distribution",
            hole=0.4
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="gender")


    with col2:

        payment = (
            original_df["PaymentMethod"]
            .value_counts()
            .reset_index()
        )

        payment.columns = ["Payment Method", "Count"]

        fig = px.bar(
            payment,
            x="Payment Method",
            y="Count",
            color="Payment Method",
            text="Count",
            title="Payment Method Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="payment")

    # ---------------- Row 2 ---------------- #

    col3, col4 = st.columns(2)

    with col3:

        fig = px.histogram(
            original_df,
            x="CourseRating",
            nbins=10,
            color_discrete_sequence=["royalblue"],
            title="Course Rating Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="rating")


    with col4:

        fig = px.histogram(
            original_df,
            x="Amount",
            nbins=20,
            color_discrete_sequence=["green"],
            title="Course Price Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="amount")


    # ---------------- Row 3 ---------------- #

    col5, col6 = st.columns(2)

    with col5:

        fig = px.histogram(
            original_df,
            x="Age",
            nbins=10,
            color_discrete_sequence=["orange"],
            title="Age Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="age")


    with col6:

        fig = px.histogram(
            original_df,
            x="CourseDuration",
            nbins=10,
            color_discrete_sequence=["purple"],
            title="Course Duration Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="duration")


    # ---------------- Row 4 ---------------- #

    col7, col8 = st.columns(2)

    with col7:

        category = (
            original_df["CourseCategory"]
            .value_counts()
            .reset_index()
        )

        category.columns = ["Category", "Count"]

        fig = px.bar(
            category,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            title="Course Category Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="category")


    with col8:

        level = (
            original_df["CourseLevel"]
            .value_counts()
            .reset_index()
        )

        level.columns = ["Level", "Count"]

        fig = px.pie(
            level,
            names="Level",
            values="Count",
            hole=0.45,
            title="Course Level Distribution"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="level")


    # ---------------- Row 5 ---------------- #

    col9, col10 = st.columns(2)

    with col9:

        fig = px.scatter(
            original_df,
            x="Age",
            y="Amount",
            color="CourseCategory",
            title="Age vs Spending"
        )

        st.plotly_chart(fig,
                        use_container_width=True,
                        key="scatter")


    with col10:

        numeric = original_df.select_dtypes(include="number")

        corr = numeric.corr()

        fig, ax = plt.subplots(figsize=(8,6))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        plt.title("Correlation Heatmap")

        st.pyplot(fig)


    # ---------------- Row 6 ---------------- #

    fig = px.histogram(
        original_df,
        x="CoursePrice",
        nbins=20,
        title="Course Price Distribution"
    )

    st.plotly_chart(fig, use_container_width=True, key="course_price")

    st.subheader("⭐ Average Course Rating by Category")

    avg_rating = (
        original_df.groupby("CourseCategory")["CourseRating"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        avg_rating,
        x="CourseCategory",
        y="CourseRating",
        color="CourseCategory",
        text_auto=".2f",
        title="Average Rating by Category"
    )

    st.plotly_chart(fig,
                    use_container_width=True,
                    key="avg_rating")
    

def learner_page():

    st.title("👤 Learner Profile Explorer")

    # Select learner by row number
    selected_index = st.selectbox(
        "Select Learner",
        processed_df.index,
        format_func=lambda x: f"Learner {x+1}"
    )

    learner = processed_df.loc[selected_index]

    # ==========================
    # Personal Information
    # ==========================

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Age", learner["Age"])

    with col2:
        st.metric("Gender", learner["Gender"])

    # ==========================
    # Learning Profile
    # ==========================

    st.subheader("Learning Profile")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric(
            "Total Courses",
            int(learner["Total_courses_enrolled"])
        )

    with col4:
        st.metric(
            "Average Spending",
            f"₹{learner['AvgSpending']:.2f}"
        )

    with col5:
        st.metric(
            "Average Rating",
            round(learner["AvgCourseRating"], 2)
        )

    col6, col7, col8 = st.columns(3)

    with col6:
        st.metric(
            "Diversity Score",
            round(learner["DiversityScore"], 2)
        )

    with col7:
        st.metric(
            "Preferred Category",
            learner["PreferredCategory"]
        )

    with col8:
        st.metric(
            "Preferred Level",
            learner["PreferredLevel"]
        )

    st.metric(
        "Enrollment Frequency",
        round(learner["Enrollment_frequency"], 2)
    )

    # ==========================
    # Cluster
    # ==========================

    cluster = learner["Cluster"]

    st.subheader("Assigned Segment")

    st.success(f"Cluster {cluster}")

    if cluster == 0:
        st.info("""
🌱 **Explorer**

This learner explores multiple categories and usually prefers beginner courses.
""")

    elif cluster == 1:
        st.info("""
🎯 **Career Focused**

This learner spends more than average and prefers career-oriented learning.
""")

    elif cluster == 2:
        st.info("""
📚 **Specialist**

This learner focuses deeply on one domain and frequently enrolls in related courses.
""")

    else:
        st.info("""
🚀 **Advanced Learner**

This learner shows a unique learning behaviour compared to other segments.
""")

    # ==========================
    # Progress Indicators
    # ==========================

    st.subheader("Learner Indicators")

    st.write("📈 Enrollment Frequency")

    progress = min(
        max(int(learner["Enrollment_frequency"] * 100), 0),
        100
    )

    st.progress(progress)

    st.write("🌍 Diversity Score")

    progress = min(
        max(int(learner["DiversityScore"] * 100), 0),
        100
    )

    st.progress(progress)

    st.write("📚 Learning Depth")

    progress = min(
        max(int(learner["Learningdepthlevel"] * 100), 0),
        100
    )

    st.progress(progress)

def cluster_page():

    st.title("📊 Cluster Visualization Dashboard")
    st.markdown("Visualize the learner segments generated using K-Means clustering.")

    col1, col2 = st.columns(2)

    with col1:

        # Cluster Distribution
            
        st.subheader("Cluster Distribution")    

        cluster_counts = processed_df["Cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Count",
            color="Cluster",
            text="Count",
            title="Learners in Each Cluster"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        # Average Spending by Cluster

        st.subheader("Average Spending by Cluster")

        cluster_spending = processed_df.groupby("Cluster")["AvgSpending"].mean().reset_index()

        fig = px.bar(
            cluster_spending,
            x="Cluster",
            y="AvgSpending",
            color="Cluster",
            title="Average Spending by Cluster"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:

        # Total Courses Enrolled by Cluster

        st.subheader("Average Courses Enrolled by Cluster")

        cluster_courses = processed_df.groupby("Cluster")["Total_courses_enrolled"].mean().reset_index()

        fig = px.bar(
            cluster_courses,
            x="Cluster",
            y="Total_courses_enrolled",
            color="Cluster",
            title="Average Courses Enrolled"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        # Preferred Course Category by Cluster

        st.subheader("Preferred Course Category by Cluster")

        fig = px.histogram(
            processed_df,
            x="PreferredCategory",
            color="Cluster",
            barmode="group",
            title="Preferred Course Category by Cluster"
        )

        st.plotly_chart(fig, use_container_width=True)    

    col5, col6 = st.columns(2)

    with col5:

        # Diversity Score by Cluster

        st.subheader("Diversity Score by Cluster")

        fig = px.box(
            processed_df,
            x="Cluster",
            y="DiversityScore",
            color="Cluster",
            title="Diversity Score Across Clusters"
        )

        st.plotly_chart(fig, use_container_width=True)    

    with col6:

        summary = processed_df.groupby("Cluster").agg({
            "Age":"mean",
            "AvgSpending":"mean",
            "Total_courses_enrolled":"mean",
            "AvgCourseRating":"mean",
            "DiversityScore":"mean"
        }).round(2)

        st.subheader("Cluster Summary")

        st.dataframe(summary)

    st.subheader("Cluster Interpretation")

    for cluster in sorted(processed_df["Cluster"].unique()):

        st.markdown(f"### Cluster {cluster}")

        temp = processed_df[processed_df["Cluster"] == cluster]

        st.write(f"Average Age : {temp['Age'].mean():.1f}")
        st.write(f"Average Spending : ₹{temp['AvgSpending'].mean():.2f}")
        st.write(f"Average Courses : {temp['Total_courses_enrolled'].mean():.1f}")    

def recommendation_page():

    st.title("📚 Personalized Course Recommendation")

    # ============================
    # Select Learner
    # ============================

    selected_index = st.selectbox(
        "Select Learner",
        processed_df.index,
        format_func=lambda x: f"Learner {x+1}"
    )

    learner = processed_df.loc[selected_index]

    cluster = learner["Cluster"]

    # ============================
    # Learner Profile
    # ============================

    st.subheader("Learner Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cluster", cluster)

    with col2:
        st.metric(
            "Preferred Category",
            learner["PreferredCategory"]
        )

    with col3:
        st.metric(
            "Preferred Level",
            learner["PreferredLevel"]
        )

    # ============================
    # Filters
    # ============================

    st.subheader("Filters")

    level = st.selectbox(
        "Course Level",
        ["All", "Beginner", "Intermediate", "Advanced"]
    )

    category = st.selectbox(
        "Course Category",
        ["All"] + sorted(original_df["CourseCategory"].unique().tolist())
    )

    # ============================
    # Recommendation Logic
    # ============================

    recommendations = original_df.copy()

    # Filter using learner preferences

    recommendations = recommendations[
        recommendations["CourseCategory"] ==
        learner["PreferredCategory"]
    ]

    recommendations = recommendations[
        recommendations["CourseLevel"] ==
        learner["PreferredLevel"]
    ]

    # User selected filters

    if category != "All":
        recommendations = recommendations[
            recommendations["CourseCategory"] == category
        ]

    if level != "All":
        recommendations = recommendations[
            recommendations["CourseLevel"] == level
        ]

    # Remove duplicate courses

    recommendations = recommendations.drop_duplicates(
        subset="CourseID"
    )

    # Highest rated first

    recommendations = recommendations.sort_values(
        by="CourseRating",
        ascending=False
    )

    recommendations = recommendations.head(5)

    # ============================
    # Display Recommendations
    # ============================

    st.subheader("🎯 Recommended Courses")

    if recommendations.empty:

        st.warning("No suitable courses found.")

    else:

        for _, row in recommendations.iterrows():

            st.markdown(f"""
### 📘 {row['CourseName']}

⭐ **Rating:** {row['CourseRating']:.1f}

📂 **Category:** {row['CourseCategory']}

🎓 **Level:** {row['CourseLevel']}

⏱ **Duration:** {row['CourseDuration']} Hours

💰 **Price:** ₹{row['Amount']}
""")

            st.divider()

    # ============================
    # Recommendation Explanation
    # ============================

    st.subheader("Why these recommendations?")

    st.info(f"""
These recommendations are generated because the learner:

- Belongs to **Cluster {cluster}**
- Prefers **{learner['PreferredCategory']}** courses
- Usually enrolls in **{learner['PreferredLevel']}** level courses
- Has an average enrolled course rating of **{learner['AvgCourseRating']:.2f}**
- Shows similar learning behaviour to other learners in the same cluster.
""")
    
def comparison_page():

    st.title("📊 Segment Comparison")

    st.write("Compare learner behaviour across different clusters.")

    # ==========================
    # Cluster Summary Table
    # ==========================

    summary = processed_df.groupby("Cluster").agg({
        "Age": "mean",
        "Total_courses_enrolled": "mean",
        "AvgSpending": "mean",
        "AvgCourseRating": "mean",
        "DiversityScore": "mean",
        "Enrollment_frequency": "mean"
    }).round(2)

    st.subheader("Cluster Summary")

    st.dataframe(summary)

    # ==========================
    # Average Spending
    # ==========================

    st.subheader("Average Spending")

    fig = px.bar(
        summary.reset_index(),
        x="Cluster",
        y="AvgSpending",
        color="Cluster",
        text="AvgSpending"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Total Courses
    # ==========================

    st.subheader("Average Courses Enrolled")

    fig = px.bar(
        summary.reset_index(),
        x="Cluster",
        y="Total_courses_enrolled",
        color="Cluster",
        text="Total_courses_enrolled"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Diversity Score
    # ==========================

    st.subheader("Diversity Score")

    fig = px.bar(
        summary.reset_index(),
        x="Cluster",
        y="DiversityScore",
        color="Cluster",
        text="DiversityScore"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Average Rating
    # ==========================

    st.subheader("Average Course Rating")

    fig = px.bar(
        summary.reset_index(),
        x="Cluster",
        y="AvgCourseRating",
        color="Cluster",
        text="AvgCourseRating"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Enrollment Frequency
    # ==========================

    st.subheader("Enrollment Frequency")

    fig = px.bar(
        summary.reset_index(),
        x="Cluster",
        y="Enrollment_frequency",
        color="Cluster",
        text="Enrollment_frequency"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # Behaviour Overview
    # ==========================

    st.subheader("Behavioural Overview")

    for cluster in sorted(processed_df["Cluster"].unique()):

        temp = processed_df[
            processed_df["Cluster"] == cluster
        ]

        st.markdown(f"### Cluster {cluster}")

        st.write(f"👥 Number of Learners : {len(temp)}")

        st.write(f"🎂 Average Age : {temp['Age'].mean():.1f}")

        st.write(f"💰 Average Spending : ₹{temp['AvgSpending'].mean():.2f}")

        st.write(f"📚 Average Courses : {temp['Total_courses_enrolled'].mean():.1f}")

        st.write(f"⭐ Average Rating : {temp['AvgCourseRating'].mean():.2f}")

        st.write(f"🌍 Diversity Score : {temp['DiversityScore'].mean():.2f}")

        st.divider()

def insights_page():

    st.title("💡 Business Insights")

    st.write(
        "Key insights generated from learner segmentation and course enrollment analysis."
    )

    # ===================================
    # Overall Metrics
    # ===================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Learners",
            len(processed_df)
        )

    with col2:
        st.metric(
            "Total Revenue",
            f"₹{original_df['Amount'].sum():,.2f}"
        )

    with col3:
        st.metric(
            "Average Rating",
            round(original_df["CourseRating"].mean(), 2)
        )

    with col4:
        st.metric(
            "Average Spending",
            f"₹{processed_df['AvgSpending'].mean():,.2f}"
        )

    st.divider()

    # ===================================
    # Most Popular Category
    # ===================================

    popular_category = (
        original_df["CourseCategory"]
        .mode()[0]
    )

    st.success(
        f"📚 Most Popular Course Category: **{popular_category}**"
    )

    # ===================================
    # Highest Spending Cluster
    # ===================================

    spending = (
        processed_df.groupby("Cluster")["AvgSpending"]
        .mean()
    )

    highest_cluster = spending.idxmax()

    st.success(
        f"💰 Cluster {highest_cluster} has the highest average spending (₹{spending.max():.2f})."
    )

    # ===================================
    # Most Active Cluster
    # ===================================

    active_cluster = (
        processed_df.groupby("Cluster")["Total_courses_enrolled"]
        .mean()
        .idxmax()
    )

    st.success(
        f"🎯 Cluster {active_cluster} has the highest average course enrollment."
    )

    # ===================================
    # Highest Rated Category
    # ===================================

    best_category = (
        original_df.groupby("CourseCategory")["CourseRating"]
        .mean()
        .idxmax()
    )

    st.success(
        f"⭐ Highest Rated Category: **{best_category}**"
    )

    st.divider()

    # ===================================
    # Recommendations
    # ===================================

    st.subheader("Recommendations for EduPro")

    st.markdown("""
- 🎯 Personalize course recommendations using learner clusters.
- 📈 Recommend courses based on learner preferences.
- 💰 Target high-spending learner segments with premium courses.
- ⭐ Promote highly-rated courses to improve engagement.
- 🔄 Encourage learners to explore new categories.
- 📊 Monitor learner behaviour continuously for better personalization.
""")

    st.divider()

    # ===================================
    # Conclusion
    # ===================================

    st.subheader("Conclusion")

    st.info("""
The EduPro learner segmentation model successfully groups learners based on their learning behaviour.

Using K-Means clustering and personalized recommendations, EduPro can improve learner engagement, increase course completion, and provide a more personalized learning experience.
""")

    # ===================================
    # Download Executive Summary
    # ===================================

    report = f"""
EduPro Student Segmentation Project

EXECUTIVE SUMMARY

Key Findings
------------
• Total Learners : {len(processed_df)}

• Total Revenue : ₹{original_df['Amount'].sum():,.2f}

• Most Popular Category : {popular_category}

• Highest Spending Cluster : {highest_cluster}

• Highest Rated Category : {best_category}

Recommendations
---------------
1. Personalize recommendations using learner clusters.
2. Promote highly-rated courses.
3. Target high-value learners.
4. Recommend courses based on preferred category and level.
5. Continuously monitor learner behaviour.

Conclusion
----------
Learner segmentation enables EduPro to provide personalized learning paths, improving engagement and learner satisfaction.
"""

    st.download_button(
        label="📥 Download Executive Summary",
        data=report,
        file_name="Executive_Summary.txt",
        mime="text/plain"
    )

with st.sidebar:
    st.title("EduPro 🎓")
    st.divider()
    selected = option_menu(
        menu_title = "Navigation",
        options = [
            "Home", "Dataset Explorer", "EDA Dashboard",
            "Learner Profile Explorer", "Cluster Visualization",
            "Course Recommendation", "Segment Comparison",
            "Business Insights"
        ],
        icons = [
            "🏡", "📊", "📈", "👥", "📌", "🎯", "📊", "💡"
        ],
        menu_icon = "mortarboard_fill",
        default_index = 0,
    )

if selected == "Home":
    home_page()
elif selected == "Dataset Explorer":
    dataset_page()
elif selected == "EDA Dashboard":
    eda_page()
elif selected == "Learner Profile Explorer":
    learner_page() 
elif selected == "Cluster Visualization":
    cluster_page()
elif selected == "Course Recommendation":
    recommendation_page()
elif selected == "Segment Comparison":
    comparison_page()
elif selected == "Business Insights":
    insights_page()    