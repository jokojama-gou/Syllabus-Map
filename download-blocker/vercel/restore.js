// public/restoration.js

// URLから画像IDを取得する関数
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

async function startRestoration() {
    const imageId = getQueryParam('id');
    const statusElement = document.getElementById('current-id');

    if (!imageId) {
        statusElement.textContent = "エラー: 表示する画像IDが指定されていません。";
        return;
    }

    statusElement.textContent = `資料 #${imageId} を復元中...`;
    document.getElementById('page-title').textContent = `Viewer #${imageId}`;

    let protectedData;
    try {
        // 【重要】動的インポート: IDに基づき適切なデータファイルをロード
        const module = await import(`./image_data_${imageId}.js`); 
        protectedData = module.protectedData;
    } catch (error) {
        statusElement.textContent = `エラー: ID ${imageId} のデータファイルが見つかりませんでした。`;
        console.error("データロードエラー:", error);
        return;
    }
    
    // 以下、元の復元ロジック
    const canvas = document.getElementById('viewer');
    const ctx = canvas.getContext('2d');

    // キャンバスサイズの計算
    const maxX = Math.max(...protectedData.map(p => p.x + p.w));
    const maxY = Math.max(...protectedData.map(p => p.y + p.h));
    canvas.width = maxX;
    canvas.height = maxY;

    const loadPromises = protectedData.map(piece => {
        return new Promise((resolve) => {
            const img = new Image();
            
            // 難読化の解除：文字列を再度「逆順」に戻す
            const originalB64 = piece.data.split("").reverse().join("");
            
            img.src = "data:image/png;base64," + originalB64;
            
            img.onload = () => {
                // 正しい座標に描画
                ctx.drawImage(img, piece.x, piece.y, piece.w, piece.h);
                resolve();
            };
            // エラー処理も追加することが望ましい
        });
    });

    // 全ピースの描画が終わったら、ウォーターマークを合成
    Promise.all(loadPromises).then(() => {
        addWatermark(canvas, ctx);
        statusElement.textContent = `資料 #${imageId} の復元が完了しました。`;
    }).catch(e => {
        console.error("画像描画中にエラーが発生しました:", e);
        statusElement.textContent = `復元中にエラーが発生しました。`;
    });

    function addWatermark(canvas, ctx) {
        ctx.save();
        ctx.globalAlpha = 0.3; 
        ctx.font = "bold 48px sans-serif";
        ctx.fillStyle = "white";
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 6); 
        ctx.textAlign = "center";
        // ユーザーの情報を使用し、ウォーターマークをパーソナライズ
        ctx.fillText("Keio SFC Archives / 横浜・舞岡", 0, 0); 
        ctx.restore();
    }

    // 右クリック対策（JSレベル）
    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        alert('この画像は保護されています。');
    });
}

startRestoration();