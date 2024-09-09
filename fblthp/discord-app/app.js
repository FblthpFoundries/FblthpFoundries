import 'dotenv/config';
import express from 'express';
import {
  InteractionType,
  InteractionResponseType,
  verifyKeyMiddleware,
} from 'discord-interactions';
import { getRandomEmoji, generateCard, DiscordRequest, sendMessageAndImage} from './utils.js';

// Create an express app
const app = express();
// Get port, or default to 3000
const PORT =  5000;


/**
 * Interactions endpoint URL where Discord will send HTTP requests
 * Parse request body and verifies incoming requests using discord-interactions package
 */
app.post('/interactions', verifyKeyMiddleware(process.env.PUBLIC_KEY), async function (req, res) {
  // Interaction type and data
  const { type, data } = req.body;

  /**
   * Handle verification requests
   */
  if (type === InteractionType.PING) {
    return res.send({ type: InteractionResponseType.PONG });
  }

  /**
   * Handle slash command requests
   * See https://discord.com/developers/docs/interactions/application-commands#slash-commands
   */
  if (type === InteractionType.APPLICATION_COMMAND) {
    const { name } = data;
    const interactionID = req.body['id']
    const appID = req.body['application_id']
    const channelID = req.body['channel_id']

    // "test" command
    if (name === 'test') {
      // Send a message into the channel where command was triggered from
      return res.send({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          // Fetches a random emoji to send from a helper function
          content: `hello world ${getRandomEmoji()}`,
        },
      });
    }

    if (name === 'curse'){

      const userID = req.body.data.options[0].value

      var msg = `<@${userID}>, Fuck you\nBitch`
      if (userID == 570371704296308736){
          msg = `<@${userID}> is the coolest and very sexy`
      }


      return res.send({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: msg,
          allowed_mentions: {
            parse: ['users']
          }
        }
      })
    }

    if (name === 'gen'){

      res.send({
        type: InteractionResponseType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE,
        data:{content: "loading"}
      })

      const card =  await generateCard(req.body.data.options)

      if(card === ''){
        await DiscordRequest(
          `webhooks/${process.env.APP_ID}/${req.body.token}`,
          {
            method:'POST',
            body: {
              content: "we made a fucky wucky"
            }
          }
        )

      }else{

        const response = await sendMessageAndImage(
          'Here\'s your card, ya filthy animal',
          card,
          `webhooks/${process.env.APP_ID}/${req.body.token}`
        )
      }

      return
       
    }

    console.error(`unknown command: ${name}`);
    return res.status(400).json({ error: 'unknown command' });
  }

  console.error('unknown interaction type', type);
  return res.status(400).json({ error: 'unknown interaction type' });
});

app.listen(PORT, () => {
  console.log('Listening on port', PORT);
});
