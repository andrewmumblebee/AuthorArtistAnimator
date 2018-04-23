// Script for scraping sprite sheets from the universal LPC-Spritesheet-Character-Generator site.
const puppeteer = require('puppeteer');
const cheerio = require('cheerio');
const request = require('request');
const GetUniqueSelector = require('cheerio-get-css-selector');
const fs = require('fs');

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
    var hair = $(options[11]).find('input');
    var combinations = [];

    // Optimization 1 - Only take screenshots of sprites not already captured in dataset.
    // Optimization 2 - Store a list of the selectors and make permutations, instead of nesting.

    sex.each(function(i) {
      let s = [$(this).getUniqueSelector() , i + 1];
      let c;
      if ($(this).next().text() == 'Male') {
        c = ['sex=female'];
      } else {
        c = ['sex=male', 'clothes=formal'];
      }
      race.each(function (i) {
        let r = [$(this).getUniqueSelector() , i + 1];
        console.log("On race" + i);
        if (!checkConstraint($(this), c)) {
          eyes.each(function (i) {
            if (Math.random() > 0.5) return true;
            let e = [$(this).getUniqueSelector() , i + 1];
            if ($(this).next().text() != 'Skeleton') {
              nose.each(function (i) {
                if (Math.random() > 0.5) return true;
                let n = [$(this).getUniqueSelector() , i + 1];
                if ($(this).next().text() != 'Skeleton') {
                  ears.each(function (i) {
                    if (Math.random() > 0.4) return true;
                    let ea = [$(this).getUniqueSelector() , i + 1];
                    legs.each(function (i) {
                      if (Math.random() > 0.4) return true;
                      let l = [$(this).getUniqueSelector() , i + 1];
                      if (!checkConstraint($(this), c)) {
                        clothes.each(function (i) {
                          if (Math.random() > 0.4) return true;
                          let cl = [$(this).getUniqueSelector() , i + 1];
                          if (!checkConstraint($(this), c)) {
                            hair.each(function(i) {
                              let h = [$(this).getUniqueSelector(), i + 1];
                              if (Math.random() > 0.3) return true;
                              if (!checkConstraint($(this), c)) {
                                let options = {'selectors' : [s[0], r[0], e[0], n[0], ea[0], l[0], cl[0], h[0]], 'labels' : `s${s[1]}_r${r[1]}_e${e[1]}_n${n[1]}_ea${ea[1]}_l${l[1]}_cl${cl[1]}_h${h[1]}` };
                                combinations.push(options);
                              }
                            });
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

    (async (combinations) => {
      const browser = await puppeteer.launch();
      const page = await browser.newPage();
      await page.goto('http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator');
      await page.setViewport({ width: 2000, height: 400 });
      const pngImage = await page.$('#spritesheet');
      console.log("Fetching sprite sheets.");

      // This has to be made synchronous, otherwise screenshot might be taken on wrong combination.
      await asyncForEach(combinations, async function(combination) {
        // Don't take a screenshot if it already exists.
        if (!fs.existsSync(`./dump/sheets/${combination['labels']}.png`)) {
          await asyncForEach(combination['selectors'], async function(selector) {
            await page.evaluate((selector) => {
              document.querySelector(selector).click();
            }, selector);
          });
          await page.waitForNavigation({waitUntil:['networkidle', 'load', 'domcontentloaded'], timeout:10000}).catch(()=>{});
          await pngImage.screenshot({
            path: `./dump/sheets/${combination['labels']}.png`,
            omitBackground: true,
            type: "png"
          });
        }
      });
      await browser.close();
    })(combinations);
  } else {
    console.log(`Couldn't successfully connect to site\n ${error}`);
  }
});

