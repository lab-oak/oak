import * as client from '@pkmn/client';
import * as data from '@pkmn/data';
import * as dex from '@pkmn/dex';
import * as engine from '@pkmn/engine';
import * as types from '@pkmn/types'

import * as https from 'https';

async function getUrlContent(url: string): Promise<string> {
    if (!url.endsWith('.log')) {
        url += '.log';
    }
    return await new Promise((resolve, reject) => {
        https.get(url, (response) => {
            let data = '';

            response.on('data', (chunk) => {
                data += chunk;
            });

            response.on('end', () => {
                resolve(data);
            });
        }).on('error', (error) => {
            reject(`Error fetching URL: ${error.message}`);
        });
    });
}

function convert(replay: client.Battle, finished: client.Battle): engine.Data<engine.Gen1.Battle> {

    let sides: engine.Gen1.Side[] = [];

    const slots: engine.Slot[] = [1, 2, 3, 4, 5, 6];

    for (const player of finished.sides) {

        const active_species = player.active[0]?.baseSpecies;

        let pokemon_arr: engine.Gen1.Pokemon[] = [];

        let i: number = 1;
        let active_index = 0;

        for (const p of player.team) {
            let statusData = { sleep: 0, self: false, toxic: 0 };
            let volatiles: engine.Gen1.Volatiles = {};

            if (p.baseSpecies === active_species) {
                volatiles.recharging = false;

                active_index = i;
            }
            const stored = {
                species: p.baseSpecies.id,
                types: [p.types[0], p.types[1] || p.types[0]] as readonly [data.TypeName, data.TypeName],
                stats: p.baseSpecies.baseStats,
                moves: p.moveSlots.map(ms => {return {id: ms.id, pp: ms.ppUsed}}),
            };

            const max_hp = p.maxhp;
            const pokemon: engine.Gen1.Pokemon = {
                species: p.baseSpecies.id,
                types: stored.types,
                level: p.level,
                hp: max_hp * p.hp / 100,
                status: p.status,
                statusData: statusData,
                stats: p.baseSpecies.baseStats,
                boosts: {
                    atk: p.boosts.atk || 0,
                    def: p.boosts.def || 0,
                    spa: p.boosts.spa || 0,
                    spe: p.boosts.spe || 0,
                    spd: p.boosts.spd || 0,
                    accuracy: p.boosts.accuracy || 0,
                    evasion: p.boosts.evasion || 0,
                },
                moves: stored.moves, // pp is missing
                volatiles: volatiles,
                stored: stored,
                position: slots[i],
            };

            pokemon_arr.push(pokemon)

            i += 1;
        }

        let side: engine.Gen1.Side = {
            active: pokemon_arr[active_index],
            pokemon: pokemon_arr,
            lastUsedMove: undefined,
            lastSelectedMove: undefined,
            lastMoveIndex: undefined,
            lastMoveCounterable: false,
        };

        sides.push(side);
    }

    return {
        sides: sides,
        turn: 0,
        lastDamage: 0,
        prng: [1, 2, 3, 4, 5, 6],
    };
}

async function getBattleFromReplayTurn(url: string, turn: number = 1000) {
    try {
        const log: string = await getUrlContent(url);
        const lines = log.split("\n");
        const gens = new data.Generations(dex.Dex);
        let finished = new client.Battle(gens);
        let current = new client.Battle(gens);

        for (const line of lines) {
            finished.add(line);
            if (current.turn < turn) {
                current.add(line);
            }
        }

        const options = {
            p1: { team: [] },
            p2: { team: [] },
            seed: [1, 2, 3, 4, 5, 6],
            showdown: true,
        };
        const restore_options = {
            p1: { name: "p1", team: [] },
            p2: { name: "p2", team: [] },
            seed: [1, 2, 3, 4, 5, 6],
            showdown: true,
            log: false,
        };

        const linter_battle = engine.Battle.create(gens.get(1), options);

        const converted_battle = convert(current, finished);

        const retore_options = {};

        const engine_battle = engine.Battle.restore(gens.get(1), linter_battle, restore_options);

    } catch (error) {
        console.error(error);
    }
}

getBattleFromReplayTurn("https://replay.pokemonshowdown.com/smogtours-gen1ou-742822", 4);