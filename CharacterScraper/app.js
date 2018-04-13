const puppeteer = require('puppeteer');
const cheerio = require('cheerio');
const request = require('request');
const GetUniqueSelector = require('cheerio-get-css-selector');

async function asyncForEach(array, callback) {
  for (let index = 0; index < array.length; index++) {
    await callback(array[index], index, array)
  }
}

function checkConstraint(element, constraints) {
  var attributes = element[0]['attribs']
  if (attributes['data-required']) {
    return constraints.some((constraint) => attributes['data-required'].includes(constraint));
  } else {
    return false;
  }
}

function skip() {
  if (Math.random() > 0.5) {
    return true;
  }
}

request('http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator', function (error, response, html) {
  if (!error && response.statusCode == 200) {
    var $ = cheerio.load(html.toString());
    GetUniqueSelector.init($);
    var options = $('#chooser > ul > li');
    var sex = $(options[0]).find('input');
    var race = $(options[1]).find('input');
    var eyes = $(options[2]).find('input');
    var nose = $(options[3]).find('input');
    var ears = $(options[4]).find('input');
    var legs = $(options[5]).find('input');
    var clothes = $(options[6]).find('input');
    var combinations = [];

    // THERE IS TOO MUCH GOING ON HERE.

    sex.each(function(i) {
      let s = [$(this).getUniqueSelector() , i];
      let c;
      if ($(this).next().text() == 'Male') {
        c = ['sex=female'];
      } else {
        c = ['sex=male', 'clothes=formal'];
      }
      race.each(function (i) {
        let r = [$(this).getUniqueSelector() , i];
        if (!checkConstraint($(this), c)) {
          eyes.each(function (i) {
            if (Math.random() > 0.5) return true;
            let e = [$(this).getUniqueSelector() , i];
            if ($(this).next().text() != 'Skeleton') {
              nose.each(function (i) {
                if (Math.random() > 0.5) return true;
                let n = [$(this).getUniqueSelector() , i];
                if ($(this).next().text() != 'Skeleton') {
                  ears.each(function (i) {
                    if (Math.random() > 0.5) return true;
                    let ea = [$(this).getUniqueSelector() , i];
                    legs.each(function (i) {
                      if (Math.random() > 0.5) return true;
                      let l = [$(this).getUniqueSelector() , i];
                      if (!checkConstraint($(this), c)) {
                        clothes.each(function (i) {
                          if (Math.random() > 0.5) return true;
                          let cl = [$(this).getUniqueSelector() , i];
                          if (!checkConstraint($(this), c)) {
                            let options = {'selectors' : [s[0], r[0], e[0], n[0], ea[0], l[0], cl[0]], 'labels' : `s${s[1]}_r${r[1]}_e${e[1]}_n${n[1]}_ea${ea[1]}_l${l[1]}_cl${cl[1]}` };
                            combinations.push(options);
                          }
                        });
                      }
                    });
                  });
                }
              });
            }
          });
        }
      });
    });

    var skips = ['Oversize', 'Skeleton']; // Don't add these to considerations.

    (async (combinations) => {
      const browser = await puppeteer.launch();
      const page = await browser.newPage();
      await page.goto('http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator');
      await page.setViewport({ width: 2000, height: 400 });
      const pngImage = await page.$('#spritesheet');
      console.log("fetching images.");
      await asyncForEach(combinations, async function(combination) {
        await asyncForEach(combination['selectors'], async function(selector) {
          await page.evaluate((selector) => {
            document.querySelector(selector).click();
          }, selector);
        });
        await page.waitFor(300);
        await pngImage.screenshot({
          path: `./dump/sheets/${combination['labels']}.png`,
          omitBackground: true,
          type: "png"
        });
      });
      await browser.close();
    })(combinations);
  }
});

