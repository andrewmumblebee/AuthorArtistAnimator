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
(async (files) => {
  let promises = [];
  for (let i = 0; i <= files.length; i++) {
    promise = new Promise((resolve) => {
      jimp.read(`${charFolder}/${files[i]}`, function (err, img) {
        img.clone().crop(0, 64, 64, 64)
        .write(`./dump/sprites/f0_${files[i]}`);
        img.clone().crop(0, 128, 64, 64)
        .write(`./dump/sprites/f1_${files[i]}`);
        img.crop(0, 256, 64, 64)
        .write(`./dump/sprites/f2_${files[i]}`);
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

// files.forEach(file => {
//   jimp.read(`${charFolder}/${file}`, function (err, img) {
//     img.crop(0, 128, 64, 64)
//     .write(`./dump/sprites/${file}`);
//   });
// });
