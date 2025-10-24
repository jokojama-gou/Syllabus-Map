import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

# --- 1. 定数とファイル名設定 ---
OUTPUT_DIR = 'sfc_kappa_maps'

# 描画設定
FILL_COLOR = (83, 83, 83, 150)  # 薄い黄色 (RGBA: 半透明)
TEXT_COLOR = (0, 1258, 255)         # 黒
FONT_SIZE = 20                 # 初期フォントサイズ (要調整)

# --- 2. 日本語フォント探索ロジック (前回と同じ) ---
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


# --- 3. データ前処理関数 (この関数を修正) ---

def preprocess_syllabus_data(df):
    """
    シラバスデータを「1行 = 1科目/1教室/1時限」の形式に分解・正規化する。
    連番時限 (例: 火2,3) を個別の時限 (火2, 火3) に分解する。
    """
    records = []
    # 欠損値を持つ行は除外
    df = df.dropna(subset=['教室', '曜日時限']).copy()

    for _, row in df.iterrows():
        
        # --- (A) 曜日時限を個別の時限コードに分解 ---
        normalized_time_slots = []
        # まず '/' で分割 (例: '火2,3/水4' -> ['火2,3', '水4'])
        time_groups = [g.strip() for g in row['曜日時限'].split('/')]
        
        for group in time_groups:
            # 正規表現で曜日と時限番号を抽出 (例: '火2,3' -> ('火', '2,3'))
            match = re.match(r'([月火水木金土日他])(\d+(,\d+)*)', group)
            
            if match:
                day = match.group(1) # 曜日
                periods_str = match.group(2) # 時限連番 (例: '2,3' または '4')
                
                # 時限連番を ',' で分割 (例: '2,3' -> ['2', '3'])
                periods = [p.strip() for p in periods_str.split(',')]
                
                for period in periods:
                    # 個別時限コードを生成 (例: '火2', '火3')
                    normalized_time_slots.append(f"{day}{period}")
        
        # --- (B) 教室情報のパースと結合 ---

        # 教室コードと時限の複合表記を抽出 (例: '火2:ο15, 水2:ι22')
        matches = re.findall(r'(\S+):(\S+)', row['教室'])
        # 複合表記内の時限コードをキーとする辞書を作成 (例: {'火2': 'ο15'})
        room_time_pairs = {time_code: room_code for room_code, time_code in matches} 

        # 時限の正規化リスト (normalized_time_slots) を使用してレコードを作成
        for ts in normalized_time_slots:
            room = None
            
            # 1. 複合表記から探す
            if ts in room_time_pairs:
                room = room_time_pairs[ts]
            else:
                # 2. 単一教室コードのみが列挙されている場合
                # ':''を含まない教室コードを抽出 (例: ['κ17'])
                simple_rooms = [r.strip() for r in row['教室'].split(',') if ':' not in r]
                
                # 単一の教室コードのみがリストにある場合、その科目の全ての時限に使用すると仮定
                if len(simple_rooms) == 1:
                    room = simple_rooms[0] 
                
                # その他の複雑なケース (複数教室・複数時限で対応が不明など) はスキップ (MVPスコープ外)

            if room and room.strip(): # 教室コードが存在し、空でないことを確認
                records.append({
                    '科目名': row['科目名'],
                    '担当者名': row['担当者名'],
                    '教室コード': room,
                    '曜日時限': ts,
                })

    return pd.DataFrame(records)

# --- 4. ファイル選択ダイアログ関数 (前回と同じ) ---
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

# --- 5. メイン処理 (前回と同じロジックで実行) ---

if __name__ == '__main__':
    print("SFC授業教室マップ生成スクリプト (κ棟MVP - 連番時限対応版) を開始します。")
    
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
    
    # 座標データの整形
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
    kappa_syllabus_df = processed_syllabus_df[
        processed_syllabus_df['教室コード'].str.startswith('κ', na=False)
    ]
    all_time_slots = kappa_syllabus_df['曜日時限'].unique()
    
    if len(all_time_slots) == 0:
        print("エラー: κ棟の授業が見つかりませんでした。")
        exit()
        
    print(f"検出されたユニークな時限: {len(all_time_slots)}件")
    print("-" * 30)

    # 5-3. フォントの準備
    try:
        initial_base_img = Image.open(base_image_path).convert("RGBA")
    except Exception as e:
        print(f"ベース画像のロード中にエラーが発生しました: {e}")
        exit()
    
    font = ImageFont.load_default()
    if GLOBAL_FONT_PATH:
        try:
            if GLOBAL_FONT_INDEX is not None:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE, index=GLOBAL_FONT_INDEX)
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} (Index: {GLOBAL_FONT_INDEX}) を使用します。")
            else:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE)
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} を使用します。")
                
        except IOError:
            print(f" -> 警告: フォントファイル '{GLOBAL_FONT_PATH}' の読み込みに失敗しました。デフォルトフォントを使用します。")
    else:
        print(" -> 警告: Windows標準の日本語フォントが見つかりませんでした。デフォルトフォントを使用するため、文字化けする可能性があります。")

    # 5-4. 全時限のループと描画
    for time_slot in sorted(all_time_slots):
        print(f"処理中: {time_slot}")

        current_schedule = kappa_syllabus_df[kappa_syllabus_df['曜日時限'] == time_slot]
        merged_df = pd.merge(current_schedule, location_df, on='教室コード', how='inner')

        if merged_df.empty:
            print(f" -> 警告: {time_slot} に κ棟の授業はありますが、座標データがありませんでした。スキップします。")
            continue

        try:
            base_img = initial_base_img.copy()
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

            # 画像の保存
            output_filename = os.path.join(OUTPUT_DIR, f"map_{time_slot.replace('/', '_')}.png")
            base_img.save(output_filename)
            print(f" -> {output_filename} を生成しました。")

        except Exception as e:
            print(f" -> エラーが発生しました ({time_slot}): {e}")
            continue

    print("-" * 30)
    print(f"処理が完了しました。生成されたマップは '{OUTPUT_DIR}' フォルダ内にあります。")