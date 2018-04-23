const jimp = require('jimp');
const fs = require('fs');
const charFolder = './dump/sheets';

function cropimage(folder, file) {
    jimp.read(`${folder}/${file}`, function (err, img) {
      img.crop(0, 64, 64, 192)
      .write(`./dump/sprites/${file}`);
    });
}

var files = fs.readdirSync(charFolder);

function cropAnimation(img, startY, frameCount, animCount, spriteId) {
  if (0.2 > Math.random()) { // Adds some randomness to what sprites have their animations cropped.
    for (let y = 0; y < animCount; y++) {
      yPos = y * 64 + startY;
      img.clone().crop(0, yPos, 64, 64)
      .write(`./dump/animations/a${yPos / 64}_${spriteId}b.png`);
    }

    for (let x = 1; x < frameCount; x++) {
      for (let y = 0; y < animCount; y++) {
        yPos = y * 64 + startY;
        img.clone().crop(x * 64, y * 64 + startY, 64, 64)
        .write(`./dump/animations/a${yPos / 64}_f${x}_${spriteId}.png`);
      }
      img.clone().crop(x * 64, 0, 64, 64)
      .write(`./dump/animations/a0_f${x}_${spriteId}.png`);
      img.clone().crop(x * 64, 64, 64, 64)
      .write(`./dump/animations/a1_f${x}_${spriteId}.png`);
      img.clone().crop(x * 64, 128, 64, 64)
      .write(`./dump/animations/a2_f${x}_${spriteId}.png`);
      img.clone().crop(x * 64, 192, 64, 64)
      .write(`./dump/animations/a3_f${x}_${spriteId}.png`);
    }
  }
}

(async (files) => {
  let promises = [];
  for (let i = 0; i <= files.length; i++) {
    promise = new Promise((resolve) => {;
      jimp.read(`${charFolder}/${files[i]}`, function (err, img) {
        img.clone().crop(0, 64, 64, 64)
        .write(`./dump/sprites/f0_${files[i]}`);
        img.clone().crop(0, 128, 64, 64)
        .write(`./dump/sprites/f1_${files[i]}`);
        img.clone().crop(0, 0, 64, 64)
        .write(`./dump/sprites/f2_${files[i]}`);

        cropAnimation(img, 0, 7, 4, i);
        cropAnimation(img, 64 * 4, 8, 4, i);
        cropAnimation(img, 64 * 8, 9, 4, i);
        cropAnimation(img, 64 * 12, 6, 4, i);
        cropAnimation(img, 64 * 16, 13, 4, i);
        cropAnimation(img, 64 * 20, 6, 1, i);

        // let n = 7; // Frame count
        // img.clone().crop(0, 0, 64, 64)
        //   .write(`./dump/animations/a0_${i}b.png`);
        //   img.clone().crop(0, 64, 64, 64)
        //   .write(`./dump/animations/a1_${i}b.png`);
        //   img.clone().crop(0, 128, 64, 64)
        //   .write(`./dump/animations/a2_${i}b.png`);
        //   img.clone().crop(0, 192, 64, 64)
        //   .write(`./dump/animations/a3_${i}b.png`);
        // for (x = 1; x < n; x++) {
        //   img.clone().crop(x * 64, 0, 64, 64)
        //   .write(`./dump/animations/a0_f${x}_${i}.png`);
        //   img.clone().crop(x * 64, 64, 64, 64)
        //   .write(`./dump/animations/a1_f${x}_${i}.png`);
        //   img.clone().crop(x * 64, 128, 64, 64)
        //   .write(`./dump/animations/a2_f${x}_${i}.png`);
        //   img.clone().crop(x * 64, 192, 64, 64)
        //   .write(`./dump/animations/a3_f${x}_${i}.png`);
        // }

        resolve();
      });
    });
    promises.push(promise);

    if (promises.length >= 5) {
      await Promise.all(promises);
      promises = [];
    }
  }
})(files)
