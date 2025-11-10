import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from rich import print
import math
# --- 変更箇所 1: ライブラリの追加 ---
from svgpathtools import svg2paths # SVG解析ライブラリ
import numpy as np # 数値計算用 (ポリゴンの中心計算で使用)


# --- 1. 定数とファイル名設定 ---
OUTPUT_DIR = 'sfc_kappa_maps'

# 描画設定
FILL_COLOR = (83, 83, 83, 150)  # 教室の塗りつぶし色
TEXT_COLOR = (0, 0, 0)         # 教室名の文字色 (黒)
FONT_SIZE = 20                 # 教室名のフォントサイズ

CUSTOM_STRING = "Yokoyama Go" 

WATERMARK_FONT_SIZE = 16       # 透かしのフォントサイズ
WATERMARK_ANGLE = -45          # 透かしの角度 (右肩上がり)
WATERMARK_COLOR = (120, 120, 120, 80) # 透かしの色 (グレー, 31%の透明度)
WATERMARK_SPACING_ZENKAKU = 15 # 透かし文字間のスペース（全角文字数）
WATERMARK_LINE_SPACING = 6     # 透かし文字の縦のスペーシング

# 曜日変換マップ
DAY_JP_TO_EN = {
    "月": "Mon", "火": "Tue", "水": "Wed", "木": "Thu", 
    "金": "Fri", "土": "Sat", "日": "Sun", "他": "etc"
}

# --- 正規化関数 (変更なし) ---
def normalize_room_code(code):
    # (中身は提供されたコードと同じなので省略)
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

# --- 2. 日本語フォント探索ロジック (変更なし) ---
def find_japanese_font():
    # (中身は提供されたコードと同じなので省略)
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

# --- 3. データ前処理関数 (変更なし) ---
def preprocess_syllabus_data(df):
    # (中身は提供されたコードと同じなので省略)
    records = []
    df = df.dropna(subset=['教室', '曜日時限']).copy()

    for _, row in df.iterrows():
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
    if not df_result.empty:
        df_result['normalize_code'] = df_result['教室コード'].apply(normalize_room_code)
    return df_result


# --- 4. ファイル選択ダイアログ関数 (変更なし) ---
def select_file(title, filetypes):
    # (中身は提供されたコードと同じなので省略)
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not file_path:
        messagebox.showerror("エラー", f"{title}が選択されませんでした。プログラムを終了します。")
        exit()
    return file_path


# ==============================================================================
# --- ▼▼▼【全体書き換え 1/2】SVG解析ロジックの差し替え ▼▼▼ ---
# ==============================================================================
def extract_room_coordinates_from_svg(svg_file_path):
    """
    SVGファイルからパス情報（教室コードと空間を定義するポリゴン頂点座標）を抽出し、DataFrameとして返す。
    パスが閉じていない場合は警告を出し、自動で閉じてから座標を抽出する。
    
    【修正点】
    存在しない `.poly()` の代わりに、`.point(t)` を使ってパスを
    N個の点群にサンプリング（線形近似）する方式に変更。
    """
    
    # パスをいくつの点（線分）で近似するか
    # 円や滑らかな曲線を表現するため、ある程度の数が必要 (例: 50)
    N_SAMPLING_POINTS = 50 

    try:
        paths, attributes = svg2paths(svg_file_path)
    except Exception as e:
        print(f"致命的エラー: SVGファイルの解析に失敗しました。ファイル形式またはライブラリの問題です: {e}")
        return pd.DataFrame()

    records = []
    
    # サンプリング用の t の値 (0.0 から 1.0 まで) を事前に生成
    # np.linspace(0, 1, N) は [0.0, ..., 1.0] を含むN個の等間隔な配列を生成
    t_values = np.linspace(0, 1, N_SAMPLING_POINTS)

    for path, attr in zip(paths, attributes):
        classroom_name = attr.get('id')
        if not classroom_name or not classroom_name.strip():
            continue

        # --- パスが閉じているかチェック (変更なし) ---
        if not path.isclosed():
            print(f"警告: パスID '{classroom_name}' は閉じていません。始点と終点を自動的に接続します。")
            try:
                path.close()
            except Exception as e:
                print(f"警告: パスID '{classroom_name}' を閉じられませんでした: {e}。このパスはスキップされます。")
                continue
        
        # --- ▼ 修正点: .poly() の代わりに .point(t) でサンプリング ▼ ---
        try:
            # .point(t) は t (0.0～1.0) の位置にあるパス上の点を複素数として返す
            # t_values (N個の t の値) を使って、N個の点をサンプリングする
            polygon_points = [path.point(t) for t in t_values]

            if not polygon_points:
                print(f"警告: パスID '{classroom_name}' から頂点を抽出できませんでした。このパスはスキップされます。")
                continue
            
            # 複素数のリストを (x, y) タプルのリストに変換 (変更なし)
            vertices = [(round(p.real), round(p.imag)) for p in polygon_points]

            records.append({
                '教室コード': classroom_name,
                'vertices': vertices
            })

        except AttributeError as ae:
            # path.point() が存在しない、というエラーは考えにくいが念のため
            print(f"致命的エラー: パスID '{classroom_name}' の処理中にライブラリ属性エラーが発生しました: {ae}。このパスはスキップされます。")
            continue
        except Exception as e:
            # その他の計算エラー
            print(f"警告: パスID '{classroom_name}' の座標サンプリング中にエラーが発生しました: {e}。このパスはスキップされます。")
            continue
        # --- ▲ 修正点ここまで ▲ ---

    if not records:
        print("致命的エラー: SVGファイルから有効な教室座標データ（ID付きパス）が見つかりませんでした。")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['normalize_code'] = df['教室コード'].apply(normalize_room_code)
    
    return df
# ==============================================================================
# --- ▲▲▲【全体書き換え 1/2】SVG解析ロジックの差し替え完了 ▲▲▲ ---
# ==============================================================================


# --- 4.5. 透かしレイヤー生成関数 (変更なし) ---
def create_watermark_layer(width, height, text_to_draw, font, color, angle, spacing_text):
    # (中身は提供されたコードと同じなので省略)
    diagonal = int(math.sqrt(width**2 + height**2))
    large_size = int(diagonal * 1.5) 
    watermark_layer = Image.new('RGBA', (large_size, large_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_layer)
    bbox = draw.textbbox((0, 0), text_to_draw + spacing_text, font=font)
    single_text_width = bbox[2] - bbox[0]
    single_text_height = (bbox[3] - bbox[1]) * WATERMARK_LINE_SPACING
    repeat_count = int(large_size / single_text_width) + 2
    full_text_line = (text_to_draw + spacing_text) * repeat_count
    for y in range(0, large_size, int(single_text_height)):
        x_offset = (y // int(single_text_height)) % 2 * int(single_text_width / 2)
        draw.text((-x_offset, y), full_text_line, font=font, fill=color)
    rotated_layer = watermark_layer.rotate(angle, resample=Image.BICUBIC)
    rotated_center_x = rotated_layer.width / 2
    rotated_center_y = rotated_layer.height / 2
    left = int(rotated_center_x - width / 2)
    top = int(rotated_center_y - height / 2)
    right = left + width
    bottom = top + height
    final_watermark = rotated_layer.crop((left, top, right, bottom))
    if final_watermark.size != (width, height):
        print(f"警告: 透かしの切り出しサイズが一致しませんでした。強制リサイズします。 Base:({width}, {height}), WM:{final_watermark.size}")
        final_watermark = final_watermark.resize((width, height), Image.LANCZOS)
    return final_watermark


# --- 5. メイン処理 ---
if __name__ == '__main__':
    print("SFC授業教室マップ生成スクリプト (κ棟MVP - SVGポリゴン対応版) を開始します。")
    
    # --- 依存ライブラリの確認 (変更なし) ---
    try:
        import svgpathtools
    except ImportError:
        print("\n[red]エラー:[/red] 'svgpathtools' ライブラリが見つかりません。")
        print("コマンドプロンプトで以下を実行してください:")
        print("  [yellow]pip install svgpathtools[/yellow]")
        exit()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 5-1. ファイル選択 (変更なし)
    print("ファイル選択ダイアログが表示されます。")
    base_image_path = select_file("1/3. ベース地図画像を選択してください", [("Image files", "*.jpg *.jpeg *.png")])
    syllabus_file_path = select_file("2/3. 授業スケジュールCSVを選択してください", [("CSV files", "*.csv")])
    location_file_path = select_file("3/3. 教室パスSVGを選択してください", [("SVG files", "*.svg")])
    
    # 5-2. データの読み込み・整形 (呼び出し先が新しい関数に入れ替わっている)
    try:
        syllabus_df = pd.read_csv(syllabus_file_path)
        # --- ここが新しい関数呼び出しになっている ---
        location_df = extract_room_coordinates_from_svg(location_file_path)
        
        if location_df.empty:
            print("[red]致命的エラー:[/red] SVG解析結果が空のため、プログラムを終了します。")
            exit()
            
    except Exception as e:
        print(f"データの読み込み中にエラーが発生しました: {e}")
        exit()
    
    print("データのパース（分解）処理中...")
    processed_syllabus_df = preprocess_syllabus_data(syllabus_df)

    if processed_syllabus_df.empty:
        print("[red]エラー:[/red] 処理対象の有効な授業データ（教室と時限が揃っているもの）が見つかりませんでした。")
        print("シラバスCSVファイルの中身を確認してください。")
        exit()

    all_time_slots = processed_syllabus_df['曜日時限'].unique() 
    
    if len(all_time_slots) == 0:
        print("エラー: κ棟の授業が見つかりませんでした。")
        exit()
        
    print(f"検出されたユニークな時限: {len(all_time_slots)}件")
    print("-" * 30)

    # 5-3. フォントと画像の準備 (変更なし)
    try:
        initial_base_img = Image.open(base_image_path).convert("RGBA")
    except Exception as e:
        print(f"ベース画像のロード中にエラーが発生しました: {e}")
        exit()
    
    font = ImageFont.load_default()
    watermark_font = ImageFont.load_default() 

    if GLOBAL_FONT_PATH:
        try:
            if GLOBAL_FONT_INDEX is not None:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE, index=GLOBAL_FONT_INDEX)
                watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, WATERMARK_FONT_SIZE, index=GLOBAL_FONT_INDEX) 
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} (Index: {GLOBAL_FONT_INDEX}) を使用します。")
            else:
                font = ImageFont.truetype(GLOBAL_FONT_PATH, FONT_SIZE)
                watermark_font = ImageFont.truetype(GLOBAL_FONT_PATH, WATERMARK_FONT_SIZE) 
                print(f"日本語フォント: {os.path.basename(GLOBAL_FONT_PATH)} を使用します。")
        except IOError:
            print(f" -> 警告: フォントファイル '{GLOBAL_FONT_PATH}' の読み込みに失敗しました。デフォルトフォントを使用します。")
    else:
        print(" -> 警告: Windows標準の日本語フォントが見つかりませんでした。デフォルトフォントを使用するため、文字化けする可能性があります。")

    spacing_text = " Yokoyama Go " * WATERMARK_SPACING_ZENKAKU


    # 5-4. 全時限のループと描画
    for time_slot in sorted(all_time_slots):
        print(f"処理中: {time_slot}")

        current_schedule = processed_syllabus_df[processed_syllabus_df['曜日時限'] == time_slot]
        merged_df = pd.merge(current_schedule, location_df, on='normalize_code', how='inner') 

        if merged_df.empty:
            print(f" -> 警告: {time_slot} に座標データがありません。透かしのみのマップを生成します。")
            # continue はしない

        try:
            # 1. 元の地図をコピー
            base_img = initial_base_img.copy()
            
            # 2. 透かしレイヤーを描画 (変更なし)
            day_jp = time_slot[0]
            period = time_slot[1:]
            day_en = DAY_JP_TO_EN.get(day_jp, day_jp)
            en_time_slot = f"{day_en} {period}"
            watermark_base_text = f"« {en_time_slot} »       {CUSTOM_STRING}       « {en_time_slot} »       "
            watermark_layer = create_watermark_layer(
                base_img.width, base_img.height,
                watermark_base_text,
                watermark_font,
                WATERMARK_COLOR,
                WATERMARK_ANGLE,
                spacing_text
            )
            
            # 3. 元画像の上に透かしを合成 (変更なし)
            base_img = Image.alpha_composite(base_img, watermark_layer)

            # 4. 合成後の画像に、教室の塗りつぶしとテキストを描画
            draw = ImageDraw.Draw(base_img)
            
            # ==============================================================================
            # --- ▼▼▼【全体書き換え 2/2】描画ロジックの修正 ▼▼▼ ---
            # ==============================================================================
            
            # merged_dfが空の場合、このループは実行されない
            for _, row in merged_df.iterrows():
                
                # 'vertices' カラムから頂点のリスト [(x1, y1), (x2, y2), ...] を取得
                vertices = row['vertices']
                text = row['科目名']
                
                # 頂点リストが空、またはNoneの場合はスキップ
                if not vertices:
                    print(f" -> 警告: 教室 '{row['教室コード']}' の頂点データが空です。スキップします。")
                    continue

                # --- 修正点A: 塗りつぶし ---
                # draw.rectangle を draw.polygon に変更
                try:
                    draw.polygon(vertices, fill=FILL_COLOR)
                except Exception as draw_e:
                    print(f" -> 警告: 教室 '{row['教室コード']}' のポリゴン描画に失敗しました: {draw_e}")
                    continue
                
                # --- 修正点B: テキスト描画 (ポリゴンの外接矩形の中心に配置) ---
                
                # 1. ポリゴンの外接矩形を計算 (Numpyを使うと高速)
                try:
                    v_array = np.array(vertices)
                    xmin = v_array[:, 0].min()
                    xmax = v_array[:, 0].max()
                    ymin = v_array[:, 1].min()
                    ymax = v_array[:, 1].max()
                except Exception as bbox_e:
                    print(f" -> 警告: 教室 '{row['教室コード']}' の中心計算に失敗しました: {bbox_e}")
                    continue
                
                # 2. 中心座標を計算
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                
                # 3. テキストサイズを取得して描画位置を決定 (ここは元のコードと同じ)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = center_x - (text_width / 2)
                text_y = center_y - (text_height / 2)
                
                draw.text((text_x, text_y), text, fill=TEXT_COLOR, font=font)

            # ==============================================================================
            # --- ▲▲▲【全体書き換え 2/2】描画ロジックの修正完了 ▲▲▲ ---
            # ==============================================================================

            # 5. 画像の保存 (変更なし)
            output_filename = os.path.join(OUTPUT_DIR, f"map_{time_slot.replace('/', '_')}.png")
            base_img.convert('RGB').save(output_filename)
            
            print(f" -> {output_filename} を生成しました。")

        except Exception as e:
            print(f" -> エラーが発生しました ({time_slot}): {e}")
            continue

    print("-" * 30)
    print(f"処理が完了しました。生成されたマップは '{OUTPUT_DIR}' フォルダ内にあります。")