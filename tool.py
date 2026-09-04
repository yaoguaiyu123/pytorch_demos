# import os
# from PIL import Image
#
# def png_to_pdf(image_path, output_pdf):
#     # Ensure the output directory exists
#     output_dir = os.path.dirname(output_pdf)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     if os.path.isdir(image_path):
#         # Process all PNG images in the folder
#         files = os.listdir(image_path)
#         images = [f for f in files if f.lower().endswith('.png')]
#         images.sort()
#         img_list = []
#         for img_file in images:
#             img_path_full = os.path.join(image_path, img_file)
#             try:
#                 with Image.open(img_path_full) as img:
#                     if img.mode == 'RGBA':  # Convert RGBA to RGB for PDF compatibility
#                         img = img.convert('RGB')
#                     img_list.append(img)
#             except Exception as e:
#                 print(f"Error processing {img_path_full}: {e}")
#         if img_list:
#             try:
#                 img_list[0].save(output_pdf, "PDF", resolution=100.0, save_all=True, append_images=img_list[1:])
#                 print(f"PDF created successfully: {output_pdf}")
#             except Exception as e:
#                 print(f"Error creating PDF: {e}")
#         else:
#             print("No images found to create PDF.")
#     elif os.path.isfile(image_path):
#         # Process a single PNG file
#         try:
#             with Image.open(image_path) as img:
#                 if img.mode == 'RGBA':  # Convert RGBA to RGB for PDF compatibility
#                     img = img.convert('RGB')
#                 img.save(output_pdf, "PDF", resolution=100.0)
#             print(f"PDF created successfully: {output_pdf}")
#         except Exception as e:
#             print(f"Error creating PDF: {e}")
#     else:
#         print("The provided image path is neither a directory nor a file.")
#         return
#
# # Example usage:
# png_to_pdf('C:/Users/吴博渊/Desktop/latex论文/assets/graphical_abstract.png',
#            'C:/Users/吴博渊/Desktop/latex论文/assets/graphical_abstract.pdf')



import numpy as np
import matplotlib.pyplot as plt

# 定义 ReLU 函数
def relu(x):
    return np.maximum(0, x)

# 定义 SiLU 函数
def silu(x):
    return x / (1 + np.exp(-x))

# 生成输入数据
x = np.linspace(-5, 5, 500)

# 计算函数值
relu_y = relu(x)
silu_y = silu(x)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(x, relu_y, label="ReLU", color="blue", linewidth=2)
plt.plot(x, silu_y, label="SiLU", color="orange", linestyle="--", linewidth=2)

# 设置坐标轴标签
plt.xlabel("x", fontsize=16)  # 放大字体
plt.ylabel("y", fontsize=16)  # 放大字体

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=12)  # x 轴刻度字体大小
plt.yticks(fontsize=12)  # y 轴刻度字体大小

# 添加辅助线
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)

# 图例与网格
plt.legend(fontsize=14)  # 图例字体大小
plt.grid(alpha=0.3)

# 显示图像
plt.show()






