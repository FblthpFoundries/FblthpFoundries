import 'dotenv/config';
import './addRequire.js' 

export const cardServer = 'http://card-server:5001'

export async function DiscordRequest(endpoint, options) {
  // append endpoint to root API URL
  const url = 'https://discord.com/api/v10/' + endpoint;
  // Stringify payloads
  if (options.body) options.body = JSON.stringify(options.body);
  // Use fetch to make requests
  const res = await fetch(url, {
    headers: {
      Authorization: `Bot ${process.env.DISCORD_TOKEN}`,
      'Content-Type': 'application/json; charset=UTF-8',
      'User-Agent': 'DiscordBot (https://github.com/discord/discord-example-app, 1.0.0)',
    },
    ...options
  });
  // throw API errors
  if (!res.ok) {
    const data = await res.json();
    console.log(res.status);
    throw new Error(JSON.stringify(data));
  }
  // return original response
  return res;
}

export async function InstallGlobalCommands(appId, commands) {
  // API endpoint to overwrite global commands
  const endpoint = `applications/${appId}/commands`;

  try {
    // This is calling the bulk overwrite endpoint: https://discord.com/developers/docs/interactions/application-commands#bulk-overwrite-global-application-commands
    await DiscordRequest(endpoint, { method: 'PUT', body: commands });
  } catch (err) {
    console.error(err);
  }
  const res = await DiscordRequest(endpoint, {method:'GET'})

  console.log(await res.json())
}

// Simple method that returns a random emoji from list
export function getRandomEmoji() {
  const emojiList = ['😭','😄','😌','🤓','😎','😤','🤖','😶‍🌫️','🌏','📸','💿','👋','🌊','✨'];
  return emojiList[Math.floor(Math.random() * emojiList.length)];
}

export function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}


export async function generateCard(options){

  var card = ''

  await fetch(cardServer + '/make_card',{
    method: 'GET',
  }).then((res) =>{ card = res.text()})

  console.log(card)

  return card

 /* var settings = ''

  const tl = (options)?options[0].value : null

  if (tl){
    settings += ` -tl \"${tl}\"`
  }

  const execSync = require('child_process').execSync

  //execSync('conda activate magic')

  const card = execSync(`python cardGenerator.py ${settings}`).toString('utf8')

  //execSync('conda deactivate')

  return card*/

}
