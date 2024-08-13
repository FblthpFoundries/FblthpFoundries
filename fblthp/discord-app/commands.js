import 'dotenv/config';
import { getRPSChoices } from './game.js';
import { capitalize, InstallGlobalCommands } from './utils.js';

// Get the game choices from game.js
function createCommandChoices() {
  const choices = getRPSChoices();
  const commandChoices = [];

  for (let choice of choices) {
    commandChoices.push({
      name: capitalize(choice),
      value: choice.toLowerCase(),
    });
  }

  return commandChoices;
}

// Simple test command
const TEST_COMMAND = {
  name: 'test',
  description: 'Basic command',
  type: 1,
  integration_types: [0, 1],
  contexts: [0, 1, 2],
};

// Command containing options
const CHALLENGE_COMMAND = {
  name: 'challenge',
  description: 'Challenge to a match of rock paper scissors',
  options: [
    {
      type: 3,
      name: 'object',
      description: 'Pick your object',
      required: true,
      choices: createCommandChoices(),
    },
  ],
  type: 1,
  integration_types: [0, 1],
  contexts: [0, 2],
};

const CURSE_COMMAND = {
  name: 'curse',
  description: 'Swears at desired user',
  options:[
    {
      type: 6,
      name: 'user',
      description: 'Pick the user to curse at',
      required: true,

    }
  ],
  type: 1,
  integration_types: [0,1],
  contexts:[0,1,2],
}

const GENERATE_COMMAND = {
  name: 'gen',
  description: 'Returns an AI generated magic card',
  options:[
    {
      type: 3,
      name: 'type_line',
      description: 'Set type line for generated card',
      required : false,
    },
  ],
  type: 1,
  integration_types: [0,1],
  contexts: [0,1,2],

}

const ALL_COMMANDS = [TEST_COMMAND, CURSE_COMMAND, GENERATE_COMMAND];

InstallGlobalCommands(process.env.APP_ID, ALL_COMMANDS);
