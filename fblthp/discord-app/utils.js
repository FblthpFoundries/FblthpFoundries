import 'dotenv/config';
import './addRequire.js' 

export const cardServer = 'http://card-server:5001'
const axios = require('axios')

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

export async function sendMessageAndImage(content, imageUrl, endpoint) {
  const url = 'https://discord.com/api/v10/' + endpoint;
  const imageRes = await fetch(imageUrl).then(res => res.blob());
  let formdata = new FormData();
  formdata.append("files[0]", imageRes, 'card.png');
  formdata.append("payload_json", JSON.stringify({
      content
  }));
  const response = await fetch(
      url, {
          method: 'post',
          body: formdata,
          headers: {
              Authorization: `Bot ${process.env.DISCORD_TOKEN}`,
          }
      }
  );
  const data = await response.json();
  return data;
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
  let type_line = ''

  if(options && options.length > 0){
    type_line = options[0]['value']
  }

  const res = await axios.post(`${cardServer}/make_card`, {text:type_line}).catch(
    err => console.log(err)
  )

  if(res.status != 200){
    return ''
  }
  
  return res.data['card']


}
