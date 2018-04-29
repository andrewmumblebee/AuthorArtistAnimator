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
    let i = 0;
    value['options'].each(function() {
      let element = $(this);
      let req = [];
      if (element.attr('data-required')) {
        let r = element.attr('data-required');
        if (r != 'clothes=gown' && r != 'clothes=formal')
          req = r.split(',');
      }
      let prohib = [];
      if (element.attr('data-prohibited')) {
        prohib = element.attr('data-prohibited').split(',');
      }

      let data = [];
      if (key == 'body' || key =='sex') {
        data = element.attr('id').replace('-', '=');
      }

        // As we use hot encoding and aritficial noise, encoding key index maps to slot of conditioning vector.
        //let parent = element.parent().parent();
        let id = element.attr('id').split(/[-_](.+)/);
        let k = id[0];
        let name = id[1].replace(/_/g, ' ');
        let label = value['label'] + (i);

        if (name != 'gown' && name != 'formal') {
          if (!encoding[key])
            encoding[key] = {};
          probabilityKeys[label] = true; // Basic way to enforce that each option appears at least once.

          choices.push([element.getUniqueSelector() , label, value['prob'], req, prohib, data]);

          encoding[key][n] = name;

          i++;
          n++;
        }
    });
    catalog.push(choices);
  }

  fs.writeFileSync('./encoding.json', JSON.stringify(encoding), 'utf8', (err) => console.log(err));

  return [catalog, probabilityKeys];
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

              if (probabilityMap[label] || arg[i][j][2] > Math.random()) {
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

request('http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/', function (error, response, html) {
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
        'options': $(options[1]).find('input').slice(9, 10),
        'prob': 1,
        'label': 'r'
      },
      'legs': {
        'options': $(options[4]).find('input'),
        'prob': 0.5,
        'label': 'l'
      },
      'clothes': {
        'options': $(options[5]).find('input'),
        'prob': 0.5,
        'label': 'c'
      },
      'eyes': {
        'options': $(options[2]).find('input'),
        'prob': 1,
        'label': 'e'
      },
      'nose': {
        'options': $(options[3]).find('input'),
        'prob': 1,
        'label': 'n'
      },
      'ears': {
        'options': $(options[28]).find('input'),
        'prob': 1,
        'label': 'o'
      },
    };

    var encoding = createJSONEncoding($, options_map);
    var catalog = encoding[0];
    var probabilityMap =  encoding[1];

    var permutations = cartesian(catalog, probabilityMap);
    console.log(`Capturing ${permutations.length} sprites.`);

    permutations = shuffle(permutations);
    const arrayColumn = (arr, n) => arr.map(x => x[n]);

    (async (permutations) => {
      const browser = await puppeteer.launch({headless: true});
      const page = await browser.newPage();
      await page.setViewport({ width: 2000, height: 400 });
      console.log("Fetching sprite sheets.");

      // This has to be made synchronous, otherwise screenshot might be taken on wrong combination.
      await asyncForEach(permutations, async function(combination) {
        await page.goto('http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/');
        // Don't take a screenshot if it already exists.
        const pngImage = await page.$('#spritesheet');
        let ids = arrayColumn(combination, 1);
        await page.evaluate((preclick) => {
          document.querySelector(preclick).click();
        }, ids[0] == 's1' ? '#clothes-gown': '#clothes-formal');

        let output_path = `./dump/sheets/${ids.join('_')}.png`;
        if (!fs.existsSync(output_path)) {
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

