const jimp = require('jimp');
const fs = require('fs');
const charFolder = './dump/sheets';
var files = fs.readdirSync(charFolder);

function cropAnimation(img, startY, frameCount, animCount, spriteId, animId) {
    for (let y = 0; y < animCount; y++) {
      yPos = y * 64 + startY;
      img.clone().crop(0, yPos, 64, 64)
      .write(`./dump/animations/a${animId + y}_${spriteId}b.png`);
    }

    let promises = [];

    for (let x = 0; x < frameCount; x++) {
      let x_ = (x + 1);
      for (let y = 0; y < animCount; y++) {
        yPos = y * 64 + startY;

        img.clone().crop(x_ * 64, y * 64 + startY, 64, 64)
        .write(`./dump/animations/a${animId + y}_f${x}_${spriteId}.png`);
      }
    }
}

(async (files) => {
  let promises = [];
  for (let i = 0; i <= files.length; i++) {
    promise = new Promise((resolve) => {
      jimp.read(`${charFolder}/${files[i]}`, function (err, img) {
        if (err)
          console.log(err);

        img.clone().crop(0, 64, 64, 64)
        .write(`./dump/sprites/f0_${files[i]}`);
        img.clone().crop(0, 128, 64, 64)
        .write(`./dump/sprites/f1_${files[i]}`);
        img.clone().crop(0, 0, 64, 64)
        .write(`./dump/sprites/f2_${files[i]}`);

        if (0.9 < Math.random()) {
          var random = Math.random();
          if (random < 1/6)
            cropAnimation(img, 0, 7, 3, i, 0);
          if (1/6 <= random && random < 1/6 * 2)
            cropAnimation(img, 64 * 4, 7, 3, i, 3);
          if (1/6 * 2 <= random && random < 1/6 * 3)
            cropAnimation(img, 64 * 8, 8, 3, i, 6);
          if (1/6 * 3 <= random && random < 1/6 * 4)
            cropAnimation(img, 64 * 12, 5, 3, i, 9);
          if (1/6 * 4 <= random && random < 1/6 * 5)
            cropAnimation(img, 64 * 16, 12, 3, i, 12);
          if (1/6 * 5 <= random)
            cropAnimation(img, 64 * 20, 5, 1, i, 15);
        }
          resolve();
      });
    });
    promises.push(promise);

    if (promises.length >= 2) {
      await Promise.all(promises);
      promises = [];
    }
  }
})(files)
