# RandomAD: A Random Kernel-based Anomaly Detector for Time Series

This is the implementation of the ECML-PKDD 2025 paper [**RandomAD: A Random Kernel-based Anomaly Detector for Time Series**](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/research/preprint_ecml_pkdd_2025_research_95.pdf).

**RandomAD** is a novel anomaly detection method for time series data that combines random convolutional kernels with kernel selection mechanism and anomaly filtering. The method can automatically select optimal window size for detection without manual parameter tuning, providing an efficient and effective approach to detecting anomalies in univariate time series.

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Jackxiini/RandomAD
cd RandomAD_code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Usage

Run RandomAD on UCR dataset:
```bash
python RandomAD.py --dataset UCR --n_kernel 1000 --rate 0.5
```

### Parameters

- `--dataset`: Dataset type (UCR, custom)
- `--n_kernel`: Number of random kernels (default: 1000)
- `--rate`: Feature selection ratio (default: 0.5)
- `--alpha`: Entropy weight in KSS (default: 0.5)
- `--beta`: Mutual information weight in KSS (default: 0.5)

### Example Output

```
Window Sizes: [10 20 30 40]
Best Window Size: 20
Time taken: 45.32 seconds
Kernel Number: 1000
Kernel selection rate: 0.5
Average Accuracy Score: 0.85
```

## 📊 Supported Datasets

- **UCR**: UCR Time Series Anomaly Detection datasets
- **Custom**: User-defined datasets (implement your own loader)

## 🎓 Citation

If you use this code in your research/work, please cite our paper:

```bibtex
Coming soon
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MiniRocket implementation based on the original paper
- UCR anomaly detection archive for benchmarking
- ECML-PKDD conference for accepting our work



