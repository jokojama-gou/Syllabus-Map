import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from rich import print
import math
from datetime import datetime

from svgpathtools import svg2paths # SVG解析ライブラリ
import numpy as np # 数値計算用
import re


# --- 1. 定数とファイル名設定 ---
OUTPUT_DIR = 'sfc_lectures_maps'

# 描画設定
FILL_COLOR = (200, 200, 200, 80)   # 教室の塗りつぶし色
TEXT_COLOR = (42, 72, 132)         # 教室名の文字色 (黒)
FONT_SIZE = 21                 # 教室名のフォントサイズ

CUSTOM_STRING = "横山 豪" 


WATERMARK_COLOR = (120,120,120,100)
WATERMARK_FONT_SIZE = 14       # 透かしのフォントサイズ
WATERMARK_ANGLE = -35         # 透かしの角度 (右肩上がり)
WATERMARK_SPACING_ZENKAKU = 15 # 透かし文字間のスペース（全角文字数）
WATERMARK_LINE_SPACING = 6 # 透かし文字の縦のスペーシング

FILL_COLOR = (200, 200, 200, 80)   # 教室の塗りつぶし色
# ★追加: 大きな透かしのフォントサイズ（画像の解像度に合わせて調整してください）
BIG_WATERMARK_FONT_SIZE = 500
BIG_WATERMARK_COLOR = (120,120,120,80)

# 曜日変換マップ
DAY_JP_TO_EN = {
    "月": "Mon", "火": "Tue", "水": "Wed", "木": "Thu", 
    "金": "Fri", "土": "Sat", "日": "Sun", "他": "etc"
}

def normalize_room_code(code):

    #教室コードを結合用に正規化する。
   # ギリシャ文字を英字に変換し、小文字化、スペース削除を行う。
    #例: "κ23" -> "kappa23", "Kappa 23" -> "kappa23", "K23" -> "kappa23"

    if not isinstance(code, str):
        return ""
    
    greek_to_latin = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
        'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
        'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
        'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',

        'Α': 'alpha', 'Β': 'beta', 'Γ': 'gamma', 'Δ': 'delta',
        'Ε': 'epsilon', 'Ζ': 'zeta', 'Η': 'eta', 'Θ': 'theta',
        'Ι': 'iota', 'Κ': 'kappa', 'Λ': 'lambda', 'Μ': 'mu',
        'Ν': 'nu', 'Ξ': 'xi', 'Ο': 'omicron', 'Π': 'pi',
        'Ρ': 'rho', 'Σ': 'sigma', 'Τ': 'tau', 'Υ': 'upsilon',
        'Φ': 'phi', 'Χ': 'chi', 'Ψ': 'psi', 'Ω': 'omega'
    }
   
    code = code.lower() 
                        
    for greek, latin in greek_to_latin.items():
        code = code.replace(greek, latin)
        
    code = re.sub(r'[\s\-_]', '', code)
    
    return code

def find_japanese_font():
    
 #  Windows環境で一般的に利用可能な日本語フォントを探してパスとインデックスを返す
    
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

    df_result = pd.DataFrame(records)
    
    # DataFrameが空でない場合 *のみ* 、正規化カラムを追加する
    if not df_result.empty:
        df_result['normalize_code'] = df_result['教室コード'].apply(normalize_room_code)

    return df_result


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

# SVG解析ロジック
def extract_room_coordinates_from_svg(svg_file_path):
    """
    SVGファイルからパス情報（教室コードとパス形状）を抽出し、DataFrameとして返す。
    ベジェ曲線を含むパスをポリゴン座標に変換して保存する。
    """
    try:
        # svg2pathsでSVGを解析。pathsはPathオブジェクトのリスト、attributesは属性のリスト
        paths, attributes = svg2paths(svg_file_path)
    except Exception as e:
        print(f"致命的エラー: SVGファイルの解析に失敗しました。ファイル形式またはライブラリの問題です: {e}")
        return pd.DataFrame()

    records = []
    
    for path, attr in zip(paths, attributes):
        # 教室コードはパスのIDとしてGIMPで設定されている
        classroom_name = attr.get('id')
        
        # IDがないパス（装飾など）はスキップ
        if not classroom_name or not classroom_name.strip():
            continue

        # パスをポリゴン座標に変換（ベジェ曲線を線分近似）
        # サンプリング数を増やすことで滑らかな曲線を再現
        try:
            # パスの長さに応じてサンプル数を決定（最小50、最大500）
            path_length = path.length()
            num_samples = max(50, min(500, int(path_length / 2)))
            
            # パスを均等にサンプリング
            polygon_points = []
            for i in range(num_samples + 1):
                t = i / num_samples
                point = path.point(t)
                polygon_points.append((point.real, point.imag))
            
            # 外接矩形も計算（テキスト配置用）
            xmin, xmax, ymin, ymax = path.bbox()
            
        except Exception as e:
            print(f"警告: パスID '{classroom_name}' の変換中にエラーが発生しました: {e}。このパスはスキップされます。")
            continue

        # データを保存
        records.append({
            '教室コード': classroom_name,
            'polygon_points': polygon_points,  # ポリゴン座標リスト
            'x_min_raw': xmin,
            'y_min_raw': ymin, 
            'x_max_raw': xmax,
            'y_max_raw': ymax,
        })

    if not records:
        print("致命的エラー: SVGファイルから有効な教室座標データ（ID付きパス）が見つかりませんでした。")
        return pd.DataFrame()

    # DataFrameに変換し、整数座標を計算（テキスト配置用の中心点）
    df = pd.DataFrame(records)
    df['x_start'] = np.floor(df['x_min_raw']).astype(int)
    df['y_start'] = np.floor(df['y_min_raw']).astype(int) 
    df['x_end'] = np.ceil(df['x_max_raw']).astype(int)
    df['y_end'] = np.ceil(df['y_max_raw']).astype(int)
    
    df['normalize_code'] = df['教室コード'].apply(normalize_room_code)
    
    # 元の生の座標列は削除
    df = df.drop(columns=['x_min_raw', 'y_min_raw', 'x_max_raw', 'y_max_raw'])

    return df


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



    # 最終確認 (万が一サイズが違ったらリサイズする)
    if final_watermark.size != (width, height):
        print(f"警告: 透かしの切り出しサイズが一致しませんでした。強制リサイズします。 Base:({width}, {height}), WM:{final_watermark.size}")
        final_watermark = final_watermark.resize((width, height), Image.LANCZOS)

    return final_watermark

# --- 5. メイン処理 ---
if __name__ == '__main__':
    print("SFC授業教室マップ生成スクリプト (κ棟MVP - SVG対応版) を開始します。")
    
    # --- 重要: 依存ライブラリの確認 ---
    try:
        import svgpathtools
    except ImportError:
        print("\n[red]エラー:[/red] 'svgpathtools' ライブラリが見つかりません。")
        print("コマンドプロンプトで以下を実行してください:")
        print("  [yellow]pip install svgpathtools[/yellow]")
        exit()
    # --- 依存ライブラリの確認終わり ---

    # 日時フォルダーの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(OUTPUT_DIR, timestamp)
    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    
    print(f"出力先: {output_subdir}")

    # 5-1. ファイル選択
    print("ファイル選択ダイアログが表示されます。")
    base_image_path = select_file("1/3. ベース地図画像を選択してください", [("Image files", "*.jpg *.jpeg *.png")])
    syllabus_file_path = select_file("2/3. 授業スケジュールCSVを選択してください", [("CSV files", "*.csv")])
    # --- 変更箇所 2: 教室座標ファイルタイプをSVGに変更 ---
    location_file_path = select_file("3/3. 教室パスSVGを選択してください", [("SVG files", "*.svg")])
    
    # 5-2. データの読み込み・整形
    try:
        syllabus_df = pd.read_csv(syllabus_file_path)
        # --- 変更箇所 3: SVG解析関数呼び出しに置き換え ---
        location_df = extract_room_coordinates_from_svg(location_file_path)
        
        if location_df.empty:
            print("[red]致命的エラー:[/red] SVG解析結果が空のため、プログラムを終了します。")
            exit()
            
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        exit()
    
    print("データのパース（分解）処理中...")
    processed_syllabus_df = preprocess_syllabus_data(syllabus_df)

    # --- ★修正★ DataFrameが空かどうかのチェックを先に行う ---
    if processed_syllabus_df.empty:
        print("[red]エラー:[/red] 処理対象の有効な授業データ（教室と時限が揃っているもの）が見つかりませんでした。")
        print("シラバスCSVファイルの中身を確認してください。")
        exit()

    # DataFrameが空でないことが保証されたので、安全に列にアクセスできる
    all_time_slots = processed_syllabus_df['曜日時限'].unique() 
    
    if len(all_time_slots) == 0:
        # このチェックは念のため残すが、通常は上の .empty でキャッチされる
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
    watermark_font = ImageFont.load_default() 
    big_watermark_font = ImageFont.load_default()

    if GLOBAL_FONT_PATH:
        try:
            if GLOBAL_FONT_INDEX is not None:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE, index=GLOBAL_FONT_INDEX)
                watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, WATERMARK_FONT_SIZE, index=GLOBAL_FONT_INDEX) 
                big_watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, BIG_WATERMARK_FONT_SIZE, index=GLOBAL_FONT_INDEX)
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} (Index: {GLOBAL_FONT_INDEX}) を使用します。")
            else:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE)
                watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, WATERMARK_FONT_SIZE) 
                big_watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, BIG_WATERMARK_FONT_SIZE)
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} を使用します。")
                
        except IOError:
            print(f" -> 警告: フォントファイル '{GLOBAL_FONT_PATH}' の読み込みに失敗しました。デフォルトフォントを使用します。")
    else:
        print(" -> 警告: Windows標準の日本語フォントが見つかりませんでした。デフォルトフォントを使用するため、文字化けする可能性があります。")

    # 透かし用の固定スペース
    spacing_text = "　" * WATERMARK_SPACING_ZENKAKU

# 5-4. 全時限のループと描画
    for time_slot in sorted(all_time_slots):
        print(f"処理中: {time_slot}")

        current_schedule = processed_syllabus_df[processed_syllabus_df['曜日時限'] == time_slot]
 
        merged_df = pd.merge(current_schedule, location_df, on='normalize_code', how='inner') 

        # merged_dfが空でも処理を続行し、透かしのみの「空のマップ」を生成する
        if merged_df.empty:
            # 警告メッセージを変更
            print(f" -> 警告: {time_slot} に座標データがありません。透かしのみのマップを生成します。")
        # 'continue' を削除し、処理を続行させる


        try:
            # 1. 元の地図をコピー
            base_img = initial_base_img.copy()
            
            # 2. 透かしレイヤーを描画 (「火5」などの情報で)
            
            # time_slot (例: "火2") を "tue 2" に変換
            day_jp = time_slot[0]
            period = time_slot[1:]
            day_en = DAY_JP_TO_EN.get(day_jp, day_jp) # マップにない場合はそのまま使用
            en_time_slot = f"{day_en} {period}"
            
            # 比率 (2/3, 1/3) に合わせてテキストを生成
            watermark_base_text = f"« {en_time_slot} »         {CUSTOM_STRING}         « {en_time_slot} »         "

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


            # 新しい透明レイヤーを作成
            big_wm_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            big_wm_draw = ImageDraw.Draw(big_wm_layer)
            
            # テキストの内容 (例: "月3")
            big_text = time_slot 
            
            # テキストサイズを取得して中央配置
            bbox = big_wm_draw.textbbox((0, 0), big_text, font=big_watermark_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x_pos = (base_img.width - text_w) / 2
            y_pos = (base_img.height - text_h) / 2
            
            # 描画 (色は既存透かしと同じ WATERMARK_COLOR を使用)
            big_wm_draw.text((x_pos, y_pos), big_text, font=big_watermark_font, fill=BIG_WATERMARK_COLOR)
            
            # ベース画像に合成 (塗りつぶしの下、背景透かしの上になる)
            base_img = Image.alpha_composite(base_img, big_wm_layer)

            # 4. 合成後の画像に、教室の塗りつぶしとテキストを描画
            draw = ImageDraw.Draw(base_img)
            
            for _, row in merged_df.iterrows():
                # ポリゴン座標を取得（ベジェ曲線がそのまま保持されている）
                polygon_points = row['polygon_points']
                text = row['科目名']
                
                # ポリゴンで塗りつぶし（ベジェ曲線の形状を保持）
                draw.polygon(polygon_points, fill=FILL_COLOR)
                
                # テキスト描画 (中央配置) - バウンディングボックスの中心を使用
                x_start, y_start, x_end, y_end = row['x_start'], row['y_start'], row['x_end'], row['y_end']
                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = center_x - (text_width / 2)
                text_y = center_y - (text_height / 2)
                
                draw.text((text_x, text_y), text, fill=TEXT_COLOR, font=font)

            # 5. 画像の保存
            output_filename = os.path.join(output_subdir, f"map_{time_slot.replace('/', '_')}.png")
            # RGBAからRGBに変換して保存 (PNGだが透明度を統合)
            base_img.convert('RGB').save(output_filename)
            
            # merged_dfが空でも「生成しました」と表示される
            print(f" -> {output_filename} を生成しました。")

        except Exception as e:
            print(f" -> エラーが発生しました ({time_slot}): {e}")
            continue

    print("-" * 30)
    print(f"処理が完了しました。生成されたマップは '{output_subdir}' フォルダ内にあります。")