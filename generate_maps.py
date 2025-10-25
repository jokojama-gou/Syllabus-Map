import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from rich import print
import math 


# --- 1. 定数とファイル名設定 ---
OUTPUT_DIR = 'sfc_kappa_maps'

# 描画設定
FILL_COLOR = (83, 83, 83, 150)  # 教室の塗りつぶし色
TEXT_COLOR = (0,1258, 255  )        # 教室名の文字色 (黒に変更)
FONT_SIZE = 20                  # 教室名のフォントサイズ

CUSTOM_STRING = "Yokoyama Go" 

WATERMARK_FONT_SIZE = 16    # 透かしのフォントサイズ
WATERMARK_ANGLE = -45         # 透かしの角度 (右肩上がり)
WATERMARK_COLOR = (120, 120, 120, 80) # 透かしの色 (グレー, 31%の透明度)
WATERMARK_SPACING_ZENKAKU = 15  # 透かし文字間のスペース（全角文字数）
WATERMARK_LINE_SPACING = 6 # 透かし文字の縦のスペーシング

# 曜日変換マップ
DAY_JP_TO_EN = {
    "月": "Mon", "火": "Tue", "水": "Wed", "木": "Thu", 
    "金": "Fri", "土": "Sat", "日": "Sun", "他": "etc"
}



# --- 2. 日本語フォント探索ロジック ---
def find_japanese_font():
    """
    Windows環境で一般的に利用可能な日本語フォントを探してパスとインデックスを返す
    """
    if 'WINDIR' not in os.environ:
        return None, None
        
    font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
    
    potential_fonts = [
        ('YuGothM.ttc', 0),   
        ('meiryo.ttc', 0),    
        ('msgothic.ttc', None) 
    ]
    
    for font_file, index in potential_fonts:
        font_path = os.path.join(font_dir, font_file)
        if os.path.exists(font_path):
            return font_path, index
            
    return None, None

GLOBAL_FONT_PATH, GLOBAL_FONT_INDEX = find_japanese_font()


# --- 3. データ前処理関数 ---
def preprocess_syllabus_data(df):
    """
    シラバスデータを「1行 = 1科目/1教室/1時限」の形式に分解・正規化する。
    """
    records = []
    df = df.dropna(subset=['教室', '曜日時限']).copy()

    for _, row in df.iterrows():
        
        # (A) 曜日時限を個別の時限コードに分解
        normalized_time_slots = []
        time_groups = [g.strip() for g in row['曜日時限'].split('/')]
        
        for group in time_groups:
            match = re.match(r'([月火水木金土日他])(\d+(,\d+)*)', group)
            
            if match:
                day = match.group(1) 
                periods_str = match.group(2) 
                periods = [p.strip() for p in periods_str.split(',')]
                
                for period in periods:
                    normalized_time_slots.append(f"{day}{period}")
        
        # (B) 教室情報のパースと結合
        matches = re.findall(r'(\S+):(\S+)', row['教室'])
        room_time_pairs = {time_code: room_code for room_code, time_code in matches} 

        for ts in normalized_time_slots:
            room = None
            
            if ts in room_time_pairs:
                room = room_time_pairs[ts]
            else:
                simple_rooms = [r.strip() for r in row['教室'].split(',') if ':' not in r]
                
                if len(simple_rooms) == 1:
                    room = simple_rooms[0] 

            if room and room.strip(): 
                records.append({
                    '科目名': row['科目名'],
                    '担当者名': row['担当者名'],
                    '教室コード': room,
                    '曜日時限': ts,
                })

    return pd.DataFrame(records)


# --- 4. ファイル選択ダイアログ関数 ---
def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw() 
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    root.destroy()
    
    if not file_path:
        messagebox.showerror("エラー", f"{title}が選択されませんでした。プログラムを終了します。")
        exit()
        
    return file_path


# ### ▼ 透かし機能追加 ▼ ###
# --- 4.5. 透かしレイヤー生成関数 ---
def create_watermark_layer(width, height, text_to_draw, font, color, angle, spacing_text):
    """
    指定されたテキストで埋め尽くされ、回転された透かしレイヤー（透明画像）を作成する
    """
    
    # 1. 回転しても全体を覆えるよう、対角線より大きいサイズの透明レイヤーを作成
    diagonal = int(math.sqrt(width**2 + height**2))
    # 余裕を持たせる
    large_size = int(diagonal * 1.5) 
    
    watermark_layer = Image.new('RGBA', (large_size, large_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)

    # 2. テキストをグリッド状に敷き詰める
    
    # 1行のテキストを作成 (画面幅より十分長く)
    bbox = draw.textbbox((0, 0), text_to_draw + spacing_text, font=font)
    single_text_width = bbox[2] - bbox[0]
    single_text_height = (bbox[3] - bbox[1]) * WATERMARK_LINE_SPACING # 行間のおおきさ
    
    # 横方向に必要な繰り返し回数
    repeat_count = int(large_size / single_text_width) + 2
    full_text_line = (text_to_draw + spacing_text) * repeat_count
    
    # 縦方向に敷き詰める
    for y in range(0, large_size, int(single_text_height)):
        # 奇数行と偶数行で開始位置をずらす
        x_offset = (y // int(single_text_height)) % 2 * int(single_text_width / 2)
        draw.text((-x_offset, y), full_text_line, font=font, fill=color)

    # 3. レイヤーを回転させる
    rotated_layer = watermark_layer.rotate(angle, resample=Image.BICUBIC)
    
    
    # ### ▼▼▼ このブロックを差し替え ▼▼▼ ###
    
    # 4. 元の画像サイズ (width, height) に合わせて中央から切り出す
    
    # 回転後の画像中心
    rotated_center_x = rotated_layer.width / 2
    rotated_center_y = rotated_layer.height / 2
    
    # 切り出す領域の計算 (整数に丸める)
    # まず左上 (left, top) を決める
    left = int(rotated_center_x - width / 2)
    top = int(rotated_center_y - height / 2)
    
    # サイズが (width, height) になるように右下 (right, bottom) を計算
    # これにより、浮動小数点の丸め誤差を防ぐ
    right = left + width
    bottom = top + height
    
    # cropメソッドには整数のタプルを渡す
    final_watermark = rotated_layer.crop((left, top, right, bottom))
    
    # ### ▲▲▲ 差し替えここまで ▲▲▲ ###


    # 最終確認 (万が一サイズが違ったらリサイズする)
    if final_watermark.size != (width, height):
        print(f"警告: 透かしの切り出しサイズが一致しませんでした。強制リサイズします。 Base:({width}, {height}), WM:{final_watermark.size}")
        final_watermark = final_watermark.resize((width, height), Image.LANCZOS)

    return final_watermark

# --- 5. メイン処理 ---
if __name__ == '__main__':
    print("SFC授業教室マップ生成スクリプト (κ棟MVP - 透かし対応版) を開始します。")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 5-1. ファイル選択
    print("ファイル選択ダイアログが表示されます。")
    base_image_path = select_file("1/3. ベース地図画像を選択してください", [("Image files", "*.jpg *.jpeg *.png")])
    syllabus_file_path = select_file("2/3. 授業スケジュールCSVを選択してください", [("CSV files", "*.csv")])
    location_file_path = select_file("3/3. 教室座標CSVを選択してください", [("CSV files", "*.csv")])
    
    # 5-2. データの読み込み・整形
    try:
        syllabus_df = pd.read_csv(syllabus_file_path)
        location_df = pd.read_csv(location_file_path).dropna(subset=['beginning_x', 'beginning_y'])
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        exit()
    
    location_df = location_df.rename(columns={
        'classroom_name': '教室コード', 'beginning_x': 'x_min', 'beginning_y': 'y_min', 'end_x': 'x_max', 'end_y': 'y_max'
    })
    location_df['x_start'] = location_df[['x_min', 'x_max']].min(axis=1).astype(int)
    location_df['x_end'] = location_df[['x_min', 'x_max']].max(axis=1).astype(int)
    location_df['y_start'] = location_df[['y_min', 'y_max']].min(axis=1).astype(int)
    location_df['y_end'] = location_df[['y_min', 'y_max']].max(axis=1).astype(int)
    location_df = location_df.drop(columns=['x_min', 'y_min', 'x_max', 'y_max'])
    
    print("データのパース（分解）処理中...")
    processed_syllabus_df = preprocess_syllabus_data(syllabus_df)
    processed_syllabus_df_syllabus_df = processed_syllabus_df.copy()
    all_time_slots = processed_syllabus_df_syllabus_df['曜日時限'].unique()    
    if len(all_time_slots) == 0:
        print("エラー: κ棟の授業が見つかりませんでした。")
        exit()
        
    print(f"検出されたユニークな時限: {len(all_time_slots)}件")
    print("-" * 30)

    # 5-3. フォントと画像の準備
    try:
        initial_base_img = Image.open(base_image_path).convert("RGBA")
    except Exception as e:
        print(f"ベース画像のロード中にエラーが発生しました: {e}")
        exit()
    
    font = ImageFont.load_default()
    watermark_font = ImageFont.load_default() # ### ▼ 透かし機能追加 ▼ ###

    if GLOBAL_FONT_PATH:
        try:
            if GLOBAL_FONT_INDEX is not None:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE, index=GLOBAL_FONT_INDEX)
                watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, WATERMARK_FONT_SIZE, index=GLOBAL_FONT_INDEX) # ### ▼ 透かし機能追加 ▼ ###
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} (Index: {GLOBAL_FONT_INDEX}) を使用します。")
            else:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE)
                watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, WATERMARK_FONT_SIZE) # ### ▼ 透かし機能追加 ▼ ###
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} を使用します。")
                
        except IOError:
            print(f" -> 警告: フォントファイル '{GLOBAL_FONT_PATH}' の読み込みに失敗しました。デフォルトフォントを使用します。")
    else:
        print(" -> 警告: Windows標準の日本語フォントが見つかりませんでした。デフォルトフォントを使用するため、文字化けする可能性があります。")

    # ### ▼ 透かし機能追加 ▼ ###
    # 透かし用の固定スペース
    spacing_text = "　" * WATERMARK_SPACING_ZENKAKU
    # ### ▲ 透かし機能追加 ▲ ###

    # 5-4. 全時限のループと描画
    for time_slot in sorted(all_time_slots):
        print(f"処理中: {time_slot}")

        current_schedule = processed_syllabus_df_syllabus_df[processed_syllabus_df_syllabus_df['曜日時限'] == time_slot]
        merged_df = pd.merge(current_schedule, location_df, on='教室コード', how='inner')

        if merged_df.empty:
            print(f" -> 警告: {time_slot} に κ棟の授業はありますが、座標データがありませんでした。スキップします。")
            continue

        try:
            # 1. 元の地図をコピー
            base_img = initial_base_img.copy()
            
            # ### ▼ 透かし機能追加 ▼ ###
            # 2. 透かしレイヤーを描画
            
            # time_slot (例: "火2") を "tue 2" に変換
            day_jp = time_slot[0]
            period = time_slot[1:]
            day_en = DAY_JP_TO_EN.get(day_jp, day_jp) # マップにない場合はそのまま使用
            en_time_slot = f"{day_en} {period}"
            
            # 比率 (2/3, 1/3) に合わせてテキストを生成
            watermark_base_text = f"« {en_time_slot} »         {CUSTOM_STRING}          « {en_time_slot} »          "

            # 透かしレイヤーを生成
            watermark_layer = create_watermark_layer(
                base_img.width, base_img.height,
                watermark_base_text,
                watermark_font,
                WATERMARK_COLOR,
                WATERMARK_ANGLE,
                spacing_text
            )
            
            # 3. 元画像の上に透かしを合成 (alpha_composite)
            base_img = Image.alpha_composite(base_img, watermark_layer)

            # 4. 合成後の画像に、教室の塗りつぶしとテキストを描画
            #    (これにより、透かしの上に教室情報が乗る)
            draw = ImageDraw.Draw(base_img)
            
            for _, row in merged_df.iterrows():
                x_start, y_start, x_end, y_end = row['x_start'], row['y_start'], row['x_end'], row['y_end']
                text = row['科目名']
                
                # 塗りつぶし
                draw.rectangle((x_start, y_start, x_end, y_end), fill=FILL_COLOR)
                
                # テキスト描画 (中央配置)
                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = center_x - (text_width / 2)
                text_y = center_y - (text_height / 2)
                
                draw.text((text_x, text_y), text, fill=TEXT_COLOR, font=font)

            # 5. 画像の保存
            output_filename = os.path.join(OUTPUT_DIR, f"map_{time_slot.replace('/', '_')}.png")
            # RGBAからRGBに変換して保存 (PNGだが透明度を統合)
            base_img.convert('RGB').save(output_filename)
            print(f" -> {output_filename} を生成しました。")

        except Exception as e:
            print(f" -> エラーが発生しました ({time_slot}): {e}")
            continue

    print("-" * 30)
    print(f"処理が完了しました。生成されたマップは '{OUTPUT_DIR}' フォルダ内にあります。")