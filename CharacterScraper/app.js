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

function checkConstraint(element, constraints) {
  var attributes = element[0]['attribs']
  if (attributes['data-required']) {
    return constraints.some((constraint) => attributes['data-required'].includes(constraint));
  } else {
    return false;
  }
}

function createJSONEncoding($, options) {
  var encoding = {};
  var catalog = [];

  var n = 0;

  for (let [key, value] of Object.entries(options)) {
    let choices = []
    //encoding[key] = {};
    // Need to check parent, if a subheading then add as parent dict
    value['options'].each(function(i) {
      let element = $(this);
      choices.push([element.getUniqueSelector() , value['label'] + (i + 1), value['prob']]);
      // As we use hot encoding and aritficial noise, encoding key index maps to slot of conditioning vector.
      //let parent = element.parent().parent();
      let id = element.attr('id').split('-');
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
  console.log(encoding);

  fs.writeFileSync('../web/static/data/encoding.json', JSON.stringify(encoding), 'utf8', (err) => console.log(err));

  return catalog;
}

function cartesian(arg) {
  var r = [], max = arg.length-1;
  function helper(arr, i) {
      for (var j=0, l=arg[i].length; j<l; j++) {
          var a = arr.slice(0); // clone arr
          if (arg[i][j][2] > Math.random()) {
            a.push(arg[i][j]);
            if (i==max)
                r.push(a);
            else
                helper(a, i+1);
          }
      }
  }
  helper([], 0);
  return r;
}


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
      'race': {
        'options': $(options[1]).find('input'),
        'prob': 1,
        'label': 'r'
      },
      'legs': {
        'options': $(options[4]).find('input'),
        'prob': 0.4,
        'label': 'l'
      },
      'clothes': {
        'options': $(options[5]).find('input'),
        'prob': 0.4,
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
      }
      // 'hair': $(options[11]).find('input'),
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

    var catalog = createJSONEncoding($, options_map);


    // var sex = $(options[0]).find('input');
    // var race = $(options[1]).find('input');
    // var eyes = $(options[2]).find('input');
    // var nose = $(options[3]).find('input');
    // var ears = $(options[4]).find('input');
    // var legs = $(options[5]).find('input');
    // var clothes = $(options[6]).find('input');
    // var hair = $(options[11]).find('input');
    var permutations = cartesian(catalog);
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

    const arrayColumn = (arr, n) => arr.map(x => x[n]);

    (async (permutations) => {
      const browser = await puppeteer.launch();
      const page = await browser.newPage();
      await page.goto('http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator');
      await page.setViewport({ width: 2000, height: 400 });
      const pngImage = await page.$('#spritesheet');
      console.log("Fetching sprite sheets.");

      // This has to be made synchronous, otherwise screenshot might be taken on wrong combination.
      await asyncForEach(permutations, async function(combination) {
        // Don't take a screenshot if it already exists.
        let output_path = `./dump/sheets/${arrayColumn(combination, 1).join('_')}.png`
        if (!fs.existsSync(output_path)) {
          await asyncForEach(arrayColumn(combination, 0), async function(selector) {
            await page.evaluate((selector) => {
              document.querySelector(selector).click();
            }, selector);
          });
          await page.waitFor(150);
          if ($('#valid-selection').text() == 1) {
            await pngImage.screenshot({
              path: output_path,
              omitBackground: true,
              type: "png"
            });
          }
        }
      });
      await browser.close();
    })(permutations);
  } else {
    console.log(`Couldn't successfully connect to site\n ${error}`);
  }
});

