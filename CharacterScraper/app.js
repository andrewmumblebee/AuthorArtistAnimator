// Script for scraping sprite sheets from the universal LPC-Spritesheet-Character-Generator site.
const puppeteer = require('puppeteer');
const cheerio = require('cheerio');
const request = require('request');
const GetUniqueSelector = require('cheerio-get-css-selector');
const fs = require('fs');

const http = require('http');
http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/html'});
  res.write(req.url);
  res.end();
}).listen(8080);

async function asyncForEach(array, callback) {
  for (let index = 0; index < array.length; index++) {
    await callback(array[index], index, array)
  }
}

function checkConstraint(constraints, data) {
  return constraints.some((constraint) => data.includes(constraint));
}

function createJSONEncoding($, options) {
  var encoding = {};
  var catalog = [];
  var probabilityKeys = {};

  var n = 0;

  for (let [key, value] of Object.entries(options)) {
    let choices = []
    //encoding[key] = {};
    // Need to check parent, if a subheading then add as parent dict
    value['options'].each(function(i) {
      let element = $(this);
      let req = [];
      if (element.attr('data-required')) {
        req = element.attr('data-required').split(',');
      }
      let prohib = [];
      if (element.attr('data-prohibited')) {
        prohib = element.attr('data-prohibited').split(',');
      }

      let data = [];
      if (key == 'body' || key =='sex') {
        data = element.attr('id').replace('-', '=');
      }

      let label = value['label'] + (i + 1);
      probabilityKeys[label] = true; // Basic way to enforce that each option appears at least once.

      choices.push([element.getUniqueSelector() , label, value['prob'], req, prohib, data]);
      // As we use hot encoding and aritficial noise, encoding key index maps to slot of conditioning vector.
      //let parent = element.parent().parent();
      let id = element.attr('id').split(/[-_](.+)/);
      let k = id[0];
      let name = id[1].replace(/_/g, ' ');
      if (!encoding[k])
        encoding[k] = {};

      encoding[k][n] = name;

      // if (parent.is("li")) {
      //   let heading = parent.first().text();
      //   console.log(element.attr('id'));
      //   encoding[key][heading] = {};
      //   encoding[key][heading][n] = element.next().text();
      // } else {
      //   encoding[key][n] = element.next().text();
      // }
      n++;
    });
    catalog.push(choices);
  }

  fs.writeFileSync('../web/static/data/encoding.json', JSON.stringify(encoding), 'utf8', (err) => console.log(err));

  return catalog, probabilityKeys;
}

function shuffle (array) {
  var i = 0
    , j = 0
    , temp = null

  for (i = array.length - 1; i > 0; i -= 1) {
    j = Math.floor(Math.random() * (i + 1))
    temp = array[i]
    array[i] = array[j]
    array[j] = temp
  }
  return array;
}

function cartesian(arg, probabilityMap) {
  var r = [], max = arg.length-1;
  function helper(arr, i, data) {
      for (var j=0, l=arg[i].length; j<l; j++) {

          let curr_data = data.concat(arg[i][j][5]);
          let req_length = arg[i][j][3].length < 1;

          if(req_length || checkConstraint(arg[i][j][3], curr_data)) { // If no conflicting required constraints.
            if(!checkConstraint(arg[i][j][4], curr_data)) { // If no conflicting prohibited constraints.
              let a = arr.slice(0); // clone arr
              let label = arg[i][j][4];

              if (probabilityMap[label]['prob'] || arg[i][j][2] > Math.random()) {
                a.push(arg[i][j]);
                if (i==max)
                    r.push(a);
                else
                    helper(a, i+1, curr_data);
              }
            }
          }
      }
  }
  helper([], 0, []);
  return r;
}

var array = [
  []
]


function handleSex(sex) {

}

request('http://127.0.0.1:8080/', function (error, response, html) {
  if (!error && response.statusCode == 200) {

    var $ = cheerio.load(html.toString());
    GetUniqueSelector.init($);
    var options = $('#chooser > ul > li');
    var options_map = {
      'sex': {
        'options': $(options[0]).find('input'),
        'prob': 1,
        'label': 's'
      },
      'body': {
        'options': $(options[1]).find('input'),
        'prob': 1,
        'label': 'r'
      },
      'legs': {
        'options': $(options[4]).find('input'),
        'prob': 0.2,
        'label': 'l'
      },
      'clothes': {
        'options': $(options[5]).find('input'),
        'prob': 0.2,
        'label': 'c'
      },
      'hair': {
        'options': $(options[27]).find('input'),
        'prob': 0.2,
        'label': 'h'
      },
      'eyes': {
        'options': $(options[2]).find('input'),
        'prob': 0.2,
        'label': 'e'
      },
      'nose': {
        'options': $(options[3]).find('input'),
        'prob': 0.2,
        'label': 'n'
      },
      'ears': {
        'options': $(options[28]).find('input'),
        'prob': 0.2,
        'label': 'o'
      },
    };

    // var options_map = {
    //   'sex': $(options[0]).find('input'),
    //   'sex': {
    //     'options': $(options[0]).find('input'),
    //     'prob': 1,
    //     'label': 's'
    //   },
    //   'race': {}
    //   'race': $(options[1]).find('input'),
    //   'eyes': $(options[2]).find('input'),
    //   'nose': $(options[3]).find('input'),
    //   'ears': $(options[4]).find('input'),
    //   'legs': $(options[5]).find('input'),
    //   'clothes': $(options[6]).find('input'),
    //   // 'hair': $(options[11]).find('input'),
    // };

    var catalog, probabilityMap = createJSONEncoding($, options_map);


    // var sex = $(options[0]).find('input');
    // var race = $(options[1]).find('input');
    // var eyes = $(options[2]).find('input');
    // var nose = $(options[3]).find('input');
    // var ears = $(options[4]).find('input');
    // var legs = $(options[5]).find('input');
    // var clothes = $(options[6]).find('input');
    // var hair = $(options[11]).find('input');
    var permutations = cartesian(catalog, probabilityMap);
    console.log(`Capturing ${permutations.length} sprites.`);

    // Optimization 1 - Only take screenshots of sprites not already captured in dataset.
    // Optimization 2 - Store a list of the selectors and make permutations, instead of nesting.

    // sex.each(function(i) {
    //   let s = [$(this).getUniqueSelector() , i + 1];
    //   let c;
    //   if ($(this).next().text() == 'Male') {
    //     c = ['sex=female'];
    //   } else {
    //     c = ['sex=male', 'clothes=formal'];
    //   }
    //   race.each(function (i) {
    //     let r = [$(this).getUniqueSelector() , i + 1];
    //     console.log("On race" + i);
    //     if (!checkConstraint($(this), c)) {
    //       eyes.each(function (i) {
    //         if (Math.random() > 0.5) return true;
    //         let e = [$(this).getUniqueSelector() , i + 1];
    //         if ($(this).next().text() != 'Skeleton') {
    //           nose.each(function (i) {
    //             if (Math.random() > 0.5) return true;
    //             let n = [$(this).getUniqueSelector() , i + 1];
    //             if ($(this).next().text() != 'Skeleton') {
    //               ears.each(function (i) {
    //                 if (Math.random() > 0.4) return true;
    //                 let ea = [$(this).getUniqueSelector() , i + 1];
    //                 legs.each(function (i) {
    //                   if (Math.random() > 0.4) return true;
    //                   let l = [$(this).getUniqueSelector() , i + 1];
    //                   if (!checkConstraint($(this), c)) {
    //                     clothes.each(function (i) {
    //                       if (Math.random() > 0.4) return true;
    //                       let cl = [$(this).getUniqueSelector() , i + 1];
    //                       if (!checkConstraint($(this), c)) {
    //                         hair.each(function(i) {
    //                           let h = [$(this).getUniqueSelector(), i + 1];
    //                           if (Math.random() > 0.3) return true;
    //                           if (!checkConstraint($(this), c)) {
    //                             let options = {'selectors' : [s[0], r[0], e[0], n[0], ea[0], l[0], cl[0], h[0]], 'labels' : `s${s[1]}_r${r[1]}_e${e[1]}_n${n[1]}_ea${ea[1]}_l${l[1]}_cl${cl[1]}_h${h[1]}` };
    //                             combinations.push(options);
    //                           }
    //                         });
    //                       }
    //                     });
    //                   }
    //                 });
    //               });
    //             }
    //           });
    //         }
    //       });
    //     }
    //   });
    // });
    permutations = shuffle(permutations);

    const arrayColumn = (arr, n) => arr.map(x => x[n]);

    (async (permutations) => {
      const browser = await puppeteer.launch({headless: true});
      const page = await browser.newPage();
      await page.setViewport({ width: 2000, height: 400 });
      console.log("Fetching sprite sheets.");

      // This has to be made synchronous, otherwise screenshot might be taken on wrong combination.
      await asyncForEach(permutations, async function(combination) {
        // Don't take a screenshot if it already exists.
        let output_path = `./dump/sheets/${arrayColumn(combination, 1).join('_')}.png`;
        if (!fs.existsSync(output_path)) {
          await page.goto('http://127.0.0.1:8080/');
          const pngImage = await page.$('#spritesheet');
          await asyncForEach(arrayColumn(combination, 0), async function(selector) {
            await page.waitForSelector(selector);
            await page.evaluate((selector) => {
              document.querySelector(selector).click();
            }, selector);
          });
          await page.waitFor(50);
          //let valid = await page.evaluate(() => document.querySelector('#valid-selection').textContent);
          await pngImage.screenshot({
            path: output_path,
            omitBackground: true,
            type: "png"
          });
        }
      });
      await browser.close();
    })(permutations);
  } else {
    console.log(`Couldn't successfully connect to site\n ${error}`);
  }
});

