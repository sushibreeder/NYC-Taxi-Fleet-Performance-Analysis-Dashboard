# NYC Taxi Analysis Dashboard

A comprehensive Streamlit-based dashboard for analyzing NYC taxi trip data and optimizing robotaxi fleet performance. This interactive dashboard provides insights into trip patterns, performance metrics, and fleet optimization strategies.

## 🚕 Project Overview

This dashboard analyzes NYC taxi trip data to provide actionable insights for fleet optimization. The project demonstrates advanced data analysis techniques including **SQL-based Exploratory Data Analysis (EDA)** using DuckDB and **causal inference** methods to identify relationships between trip factors and performance outcomes. It includes performance metrics, trip analysis, and data-driven recommendations for improving operational efficiency in the taxi industry.

## ✨ Features

- **SQL-Based EDA**: Advanced exploratory data analysis using DuckDB SQL queries
- **Causal Inference Analysis**: Statistical methods to identify cause-and-effect relationships
- **Trip Performance Analysis**: Average trip duration, distance, and fare analysis
- **Fleet Optimization Insights**: Data-driven recommendations for fleet management
- **Interactive Visualizations**: Dynamic charts and graphs for data exploration
- **Real-time Data Processing**: Live data analysis using DuckDB
- **Responsive Design**: Mobile-friendly interface built with Streamlit
- **Performance Metrics**: Key performance indicators for fleet operations

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **DuckDB**: In-memory analytical database for SQL-based EDA
- **SQL**: Primary language for data exploration and analysis
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **Statistical Analysis**: Causal inference and correlation analysis
- **Git**: Version control

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd taxi_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard Locally
```bash
streamlit run dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## 📊 Usage

### Dashboard Sections

1. **Project Overview**: Summary statistics and key metrics
2. **SQL-Based EDA**: Interactive SQL queries and data exploration
3. **Causal Inference Analysis**: Statistical relationships and correlations
4. **Trip Analysis**: Detailed trip performance data
5. **Fleet Optimization**: Data-driven recommendations and insights
6. **Performance Metrics**: KPIs and operational data

### How to Use

- Navigate through different sections using the sidebar
- **Run SQL queries** for custom data exploration using DuckDB
- **Analyze causal relationships** between trip factors and outcomes
- Interact with charts and graphs for detailed analysis
- View performance metrics and optimization recommendations
- Export data and insights for further analysis

## 📁 Project Structure

```
taxi_project/
├── dashboard.py          # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
└── data/                # Data files (if any)
```

## 🌐 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
3. **Deploy**: Streamlit Cloud will automatically deploy your app
4. **Access**: Your dashboard will be available at a public URL

### Live Demo
🚀 **Live Dashboard**: [NYC Taxi Fleet Performance Analysis Dashboard](https://nyc-taxi-fleet-performance-analysis-dashboard-m9srulrllkeivnsa.streamlit.app/)

Visit the live dashboard to see it in action!

## 📈 Data Sources

The dashboard currently uses sample data for demonstration purposes. In production, it can be connected to:
- NYC Taxi & Limousine Commission (TLC) data
- Real-time trip data feeds
- Historical performance databases
- Fleet management systems

## 🔧 Development

### Local Development Setup

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test locally**: `streamlit run dashboard.py`
5. **Commit and push**: 
   ```bash
   git add .
   git commit -m "Add your feature description"
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Include docstrings for functions

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for data insights
- Streamlit team for the amazing framework
- DuckDB for fast analytical queries
- Open source community for inspiration

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the documentation
- Review the error logs

---

**Happy analyzing! 🚕📊**
