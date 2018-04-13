const jimp = require('jimp');
const fs = require('fs');
const charFolder = './dump/sheets';

function cropimage(folder, file) {
    jimp.read(`${folder}/${file}`, function (err, img) {
      img.crop(0, 128, 64, 64)
      .write(`./dump/sprites/${file}`);
    });
}
var files = fs.readdirSync(charFolder);
(async (files) => {
  for (let i = 0; i <= files.length; i++) {
    await new Promise((resolve) => {
      jimp.read(`${charFolder}/${files[i]}`, function (err, img) {
        img.crop(0, 128, 64, 64)
        .write(`./dump/sprites/${files[i]}`);
        resolve();
      });
    });
  }
})(files)

// files.forEach(file => {
//   jimp.read(`${charFolder}/${file}`, function (err, img) {
//     img.crop(0, 128, 64, 64)
//     .write(`./dump/sprites/${file}`);
//   });
// });
