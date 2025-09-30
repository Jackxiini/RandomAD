# RandomAD: A Random Kernel-based Anomaly Detector for Time Series

[![ECML-PKDD](https://img.shields.io/badge/Published%20in-ECML--PKDD-blue)](https://2024.ecmlpkdd.org/)

**RandomAD** is a novel anomaly detection method for time series data that combines random kernel features with mutual information-based feature selection. This implementation provides an efficient and effective approach to detecting anomalies in univariate time series.

## 📖 Abstract

RandomAD leverages the power of random convolutional kernels (MiniRocket) combined with a novel feature selection strategy based on mutual information and entropy. The method automatically selects the most informative features while maintaining computational efficiency, making it suitable for real-time anomaly detection applications.

## 🚀 Features

- **Random Kernel Features**: Utilizes MiniRocket for fast feature extraction
- **Intelligent Feature Selection**: KSS (Kernel Selection Score) based on mutual information and entropy
- **Automatic Window Size Selection**: Adaptive window sizing based on autocorrelation analysis
- **Multiple Dataset Support**: Compatible with UCR, NAB, MSDS, SWaT, and custom datasets
- **Efficient Implementation**: Parallel processing with joblib for large-scale data

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RandomAD_code
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## 🎯 Usage

### Basic Usage

Run RandomAD on UCR dataset:
```bash
python3 RandomAD.py --dataset UCR --n_kernel 1000 --rate 0.5 --alpha 0.5 --beta 0.5 --k_neighbors 3
```

### Parameters

- `--dataset`: Dataset type (UCR, custom)
- `--n_kernel`: Number of random kernels (default: 1000)
- `--rate`: Feature selection ratio (default: 0.5)
- `--alpha`: Entropy weight in KSS (default: 0.5)
- `--beta`: Mutual information weight in KSS (default: 0.5)
- `--k_neighbors`: Number of neighbors for KNN (default: 3)

### Example Output

```
=== Package Versions ===
numpy: 1.21.0
scikit-learn: 1.0.2
pandas: 1.3.3
joblib: 1.1.0
scipy: 1.7.1
numba: 0.54.1
========================

Window Sizes: [10 20 30 40]
Best Window Size: 20
Time taken: 45.32 seconds
Kernel Number: 1000
Kernel selection rate: 0.5
Average Accuracy Score: 0.85
```

## 📁 Project Structure

```
RandomAD_code/
├── RandomAD.py          # Main execution script
├── randomad_core.py     # Core algorithm implementation
├── minirocket.py        # MiniRocket implementation
├── utils.py             # Dataset configurations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔬 Algorithm Overview

1. **Feature Extraction**: Apply MiniRocket to extract random kernel features
2. **Feature Selection**: Use KSS score combining entropy and mutual information
3. **Window Selection**: Automatically determine optimal window size
4. **Anomaly Detection**: Apply KNN on selected features to detect anomalies

## 📊 Supported Datasets

- **UCR**: UCR Time Series Anomaly Detection datasets
- **Custom**: User-defined datasets (implement your own loader)

## 🎓 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{randomad2024,
  title={RandomAD: A Random Kernel-based Anomaly Detector for Time Series},
  author={[Authors]},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please contact [your-email@domain.com].

## 🙏 Acknowledgments

- MiniRocket implementation based on the original paper
- UCR dataset for benchmarking
- ECML-PKDD conference for accepting our work
