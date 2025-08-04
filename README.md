# REAL\_ESRGAN-minimal-inference

A minimal implementation for REAL-ESRGAN inference, with PyTorch as the only core dependency.

轻量版 REAL-ESRGAN 推理实现，仅需 PyTorch 即可运行。

## 🌟 简介 (Introduction)

本项目提供 REAL-ESRGAN 的最小化推理方案，剥离冗余依赖，专注于核心超分功能，适合需要轻量集成或快速验证的场景。

This project provides a minimal inference solution for REAL-ESRGAN, stripping redundant dependencies and focusing on core super-resolution functionality. It is suitable for scenarios requiring lightweight integration or rapid validation.

## 🚀 安装 (Installation)

仅需 PyTorch 环境，无需复杂依赖：

Only requires a PyTorch environment, no complex dependencies:



```bash 
# 安装核心依赖

# Install core dependencies

pip install torch torchvision
```

## 📖 使用方法 (Usage)



1.  克隆仓库并进入目录：

    Clone the repository and enter the directory:



```bash 
git clone https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference.git

cd REAL_ESRGAN-minimal-inference
```



1.  运行推理示例：

    Run the inference example:



```bash 
# 可以直接参考 run.sh
# single image
python inference.py --input input_image.jpg --output output_path --model_name RealESRGAN_x2
# folder
python inference.py --input input_image_path --output output_path --model_name RealESRGAN_x2
```

## 🧩 模型支持 (Model Support)

目前仅支持少量常用模型(如 `RealESRGAN_x2`、`realesrgan_x4plus_anime_6B`)，主要覆盖基础超分/动漫场景。

Currently, only a few commonly used models are supported (e.g., `RealESRGAN_x2`, `realesrgan_x4plus_anime_6B`), mainly covering basic super-resolution/anime scenarios.

若你需要其他模型支持(如特定缩放倍数、轻量化模型等)，欢迎在 [Is](https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/issues)[sues](https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/issues) 中提出需求，我们会根据反馈优先适配。

If you need support for other models (such as specific scaling factors, lightweight models, etc.), please feel free to submit your requirements in [Issu](https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/issues)[es](https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/issues). We will prioritize adaptation based on feedback.

## ⭐ 支持与贡献 (Support & Contribution)



*   如果你觉得本项目有帮助，欢迎点亮右上角的 ⭐ Star 支持！

    If you find this project helpful, please click the ⭐ Star in the top-right corner to support us!

*   遇到问题或有功能建议，可通过 [Issues](https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/issues) 反馈

    For issues or feature suggestions, please provide feedback via [Issues](https://github.com/Holasyb918/REAL_ESRGAN-minimal-inference/issues)


## 🙏 致谢 (Acknowledgements)

基于 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 原始项目的核心算法实现，感谢原作者团队的开源贡献。

Based on the core algorithm implementation of the original [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) project. Thanks to the original author team for their open-source contributions.
