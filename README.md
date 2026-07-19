# 🎓 EduPro: Student Segmentation and Personalized Course Recommendation System

## 📖 Overview

EduPro is a machine learning-based web application that segments learners based on their learning behavior and provides personalized course recommendations.

The project applies data preprocessing, feature engineering, Principal Component Analysis (PCA), K-Means clustering, and recommendation techniques to improve learner engagement and the online learning experience.

---

## 🔗 Project Links

🌐 **Live Streamlit App**  
https://your-streamlit-app-url.streamlit.app

💻 **GitHub Repository**  
https://github.com/purple279/ML_Internship_Projects

🎥 **Project Demo (YouTube)**  
https://youtu.be/your-video-link

📄 **Research Paper**  
Research_paper/EduPro%20Research%20paper.pdf

## 🚀 Features

- Interactive Streamlit Dashboard
- Exploratory Data Analysis (EDA)
- Learner Profile Explorer
- Student Segmentation using K-Means Clustering
- Hierarchical Cluster Visualization
- Personalized Course Recommendation
- Business Insights Dashboard

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

---

## 📂 Project Structure

```
ML_Internship_Projects
│
├── Model/
│   ├── feature_columns.pkl
│   ├── kmeans_model.pkl
│   ├── pca.pkl
│   └── scaler.pkl
│
├── Research_paper/
│   ├── EduPro Research paper.pdf
│   └── Student_Segmentation_and_Personalized_Course_Recommendation_System.pdf
│
├── streamlitapp.py
├── README.md
└── requirements.txt
```

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Feature Selection
5. Feature Scaling
6. Principal Component Analysis (PCA)
7. K-Means Clustering
8. Hierarchical Clustering
9. Personalized Course Recommendation
10. Interactive Streamlit Dashboard

---

## 📌 Key Features Created

- Total_courses_enrolled
- AvgCoursesPerCategory
- Enrollment_frequency
- PreferredCategory
- PreferredLevel
- AvgCourseRating
- AvgSpending
- DiversityScore
- LevelScore
- Learningdepthlevel
- Age_group
- HighlyRated
- ExcellentCourse
- LongDurationCourse
- ShortDurationCourse
- TotalSpent

---

## 📈 Clustering

- Algorithm: K-Means Clustering
- Dimensionality Reduction: PCA
- Cluster Selection: Elbow Method & Silhouette Score

---

## 🎯 Recommendation Logic

Recommendations are generated based on:

- Preferred Category
- Preferred Level
- Course Rating
- Course Category
- Excluding previously enrolled courses
- Top-5 highest-ranked courses

---

## 📱 Dashboard Pages

- Home
- Dataset Explorer
- EDA Dashboard
- Learner Profile Explorer
- Cluster Visualization
- Course Recommendation
- Segment Comparison
- Business Insights

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/purple279/ML_Internship_Projects.git
```

Install the required packages

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run streamlitapp.py
```

---

## 👩‍💻 Author

**Vashundthera S**

B.E. Computer Science and Engineering

Sona College of Technology

Salem, Tamil Nadu, India

---

## 📄 Research Paper

The research paper is available in the **Research_paper** folder.

---

## 📜 License

This project was developed as part of the Unified Mentor Machine Learning Internship for educational purposes.
