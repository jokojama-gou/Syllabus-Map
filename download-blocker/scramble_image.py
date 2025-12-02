import base64
import json
import random
from PIL import Image
import io
import os
import tkinter as tk
from rich import print



# 設定
INPUT_DIR = "source_images"  # 元画像があるディレクトリ
OUTPUT_DIR = "public"        # 出力先のディレクトリ
GRID_SIZE = 4

def generate_obfuscated_data(input_path, output_id):
    """単一の画像を処理し、難読化されたJSファイルを出力する"""
    
    # 元画像のロード
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"エラー: {input_path} が見つかりません。")
        return None

    w, h = img.size
    piece_w = w // GRID_SIZE
    piece_h = h // GRID_SIZE
    pieces = []

    # (元のコードと同じ難読化処理... 分割、Base64エンコード、文字列反転)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # ... (分割・エンコード処理は元のコードの通り)
            left = col * piece_w
            top = row * piece_h
            box = (left, top, left + piece_w, top + piece_h)
            piece_img = img.crop(box)

            buffer = io.BytesIO()
            piece_img.save(buffer, format="PNG")
            binary_data = buffer.getvalue()
            b64_str = base64.b64encode(binary_data).decode('utf-8')

            # 難読化：Base64文字列を「逆順」にする
            obfuscated_str = b64_str[::-1]

            pieces.append({
                "id": f"{row}-{col}",
                "data": obfuscated_str,
                "x": left,
                "y": top,
                "w": piece_w,
                "h": piece_h
            })

    # データの順序をシャッフル
    random.shuffle(pieces)
    
    # ファイル名と内容の定義
    output_js = os.path.join(OUTPUT_DIR, f"image_data_{output_id}.js")
    js_content = f"export const protectedData = {json.dumps(pieces, indent=2)};"
    
    # JSファイルとして書き出し
    with open(output_js, "w", encoding="utf-8") as f:
        f.write(js_content)
    
    print(f"完了: ID {output_id} のデータ ({len(pieces)}個の断片) を作成しました。")
    return {"id": output_id, "file_name": os.path.basename(input_path), "width": w, "height": h}

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_list = []
    
    # source_imagesディレクトリ内の全画像ファイル (.jpg/.png) を処理
    for i, file_name in enumerate(sorted(os.listdir(INPUT_DIR))):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # IDをファイル名からではなく、連番で振る（シンプルにするため）
            # ファイル名からIDを取得したい場合はここを調整
            image_id = i + 1 
            input_path = os.path.join(INPUT_DIR, file_name)
            
            result = generate_obfuscated_data(input_path, image_id)
            if result:
                image_list.append(result)

    # 全画像リストをJSONとして出力（インデックスページ用）
    list_js_content = f"export const imageList = {json.dumps(image_list, indent=2)};"
    list_output = os.path.join(OUTPUT_DIR, "image_list.js")
    with open(list_output, "w", encoding="utf-8") as f:
        f.write(list_js_content)
    
    print(f"\n全 {len(image_list)} 個の画像の処理が完了しました。")

if __name__ == "__main__":
    main()