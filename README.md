### `Overview` 

- Autonomous agents each follow simple local rules (sense, steer, deposit), and the image emerges from their collective behaviour.
- Shortcuts:

          | Key | Action |
          |---|---|
          | Space | Play / pause |
          | R | Reset |
          | C | Clear |
          | N | New seed |
          | P | Random palette |
          | Ctrl+S | Export PNG |
          | Ctrl+O | Load preset / PNG |
          | Ctrl+E | Evolve |
          | F2 | Agent activity panel |
          | A | Agent overlay on canvas |
          | F11 | Fullscreen |

- Exported PNGs embed their settings, so loading a PNG restores the exact rules that made it.

<img width="1480" height="900" alt="image_test" src="https://github.com/user-attachments/assets/552b6e74-82c7-4643-8249-5c3a0f74b302" />


### `Install and run` 

        python -m venv .venv

- Windows: .venv\Scripts\activate    
- Linux: source .venv/bin/activate

        pip install -r requirements.txt

        python -m swarmart

- Or install as a package (`pip install .`) and run_`swarmart`.


### `Agent models` 

        | Model | Local rule | Emergent result |
        |---|---|---|
        | **Physarum**|Each agent samples a chemical trail with three sensors, turns toward the strongest, moves, deposits; the trail diffuses and decays | Transport networks, veins, branching filaments |
        | **Boids** |Alignment, cohesion and separation computed from an O(N) spatial-hash mean field | Flocks, ribbons, murmurations |
        | **Flow field**|Agents steer along a divergence free curl noise field (∇×ψ of fractal periodic noise), with finite lifetimes | Silky streamlines, marbling |
        | **Turmites**|Generalized Langton's ants (curated RL rule strings) reading and rewriting a shared grid | Highways, symmetric growth, crystal fills |

- Optionally, click or drag on the canvas to drop a stimulus the swarm reacts to.

### `Fully autonomous by default` 


- Start the app and leave it alone. No clicks are needed: the swarm paints, the program decides when each piece is finished, saves it, and starts the next one.

1. **Painting.** Agents follow their local rules and the image emerges live on the canvas.
2. **Completion.** A piece is finished when it converges, reaches its step target, or hits the per-piece time limit (180 s by default; set it under *Autonomous gallery*).
3. **Autosave and scoring.** The finished piece is saved at full resolution to `Pictures/SwarmArt Gallery` as `swarmart_auto_*.png`, with its settings embedded. It is then scored by the aesthetic model. Only the newest 500 autosaves are kept, and only files with that prefix in that folder are ever deleted.
4. **Learning.** An autonomous curator learns which models produce the best pieces. It uses a UCB1-Tuned multi armed bandit whose reward is 85% aesthetic score and 15% novelty (distance to the nearest earlier pieces in a 6-D style-descriptor space). It explores every model once, never paints the same model twice in a row, and remembers what it learned between sessions in `curator.json` in the app data folder.
5. **Tuning.** While a piece paints, the next piece's rules are tuned by CMA-ES in a separate process. That process never competes with painting for Python's GIL. If it crashes it is detected and restarted, and the next piece falls back to the untuned proposal. At most one tuning task is in flight.
6. **Next piece.** After a 5-second hold, the next piece starts by itself. The panel shows pieces finished, the last and best scores, what's up next, whether its rules are AI-tuned yet, and a countdown.

- Untick *Paint, save and start new pieces on its own* to switch to manual runs. **New run** skips straight to the curator's next piece. On Linux and macOS, SIGTERM and Ctrl+C close the app cleanly.

### `Live agent activity` 

- The **Agent activity** panel (F2) and the canvas overlay (A) show, in real time, which agents are working and what they are producing.

- **Status of every agent.** Every agent that deposits paint counts as painting. It is *drawing new strokes* when its deposits land in dark, unpainted parts of the picture (below 35% of the displayed brightness, judged against the same scale the renderer uses), *reworking existing strokes* when it paints over bright existing paint (for turmites, erasing counts as reworking), and *idle* when it isn't depositing. On an empty canvas every agent starts out drawing new.
- **Canvas overlay.** A stable sample of up to 3,000 agents is drawn live, coloured by status. Agents drawing new strokes are shown as comets: an outlined head plus a short tail along their motion, so you can watch where new content is appearing. The legend shows live counts. After about 60 steps, the five agents that have drawn the most new strokes are ringed and labelled with their id.
- **Behaviour breakdown.** Each model reports what its agents are doing:
  - Physarum: following, steering, exploring
  - Boids: flocking, gathering, avoiding, wandering
  - Flow: streaming, spawning, fading
  - Turmites: building, erasing
- **Progress to completion.** A progress bar with an ETA, step count, elapsed time, the share of the canvas painted, new pixels per step, and a live sparkline.
- **Completion.** A run is complete when one of these happens:
  - **Converged:** new territory has dropped below 3% of its peak *and* a time smoothed thumbnail of the image has stopped changing, sustained over 100 steps. Moving swarms (canvas fade on) also need a warm-up proportional to the fade time.
  - **Step target reached:** you set a target in the panel.
  - **Automatic step limit reached:** a per-model safety cap for swarms that never settle, such as flocks and chaotic turmites.
- **When it completes.** The run pauses (optional), the panel shows the reason, the taskbar flashes, and Export / Continue / New run become available.

- Telemetry costs about 7% of simulation time on large swarms, and the overlay geometry is built off the UI thread in under 1 ms.

### `AI ruler` 

- **Evolve rules** runs CMA-ES (full covariance, rank-μ and rank one updates, cumulative step size adaptation, box handling by reflection) over each model's rule parameters plus palette. 
- Every candidate is simulated on a small preview and scored with an aesthetic model built from:

- luminance entropy
- coverage
- radial power-spectrum slope (target ≈ 1/f², like natural images)
- box-counting fractal dimension of the edge map
- Hasler–Süsstrunk colourfulness
- hue variety (circular variance of opponent-colour angles; this stops the optimizer gaming colourfulness with one saturated hue)

- To avoid false positives, fitness uses common random seeds within a generation. 
- The best four candidates are then re-scored on held out seeds, and evolved settings are applied only if they beat the baseline on that validation.

### `Reliability` 

- **Threads.** Simulation and rendering run on their own threads with frame back-pressure, so the UI never queues stale frames.
- **Large canvases.** Above about 1.1 MP the live view is box downsampled; export always renders at full resolution.
- **Health monitor.** Repairs NaN or ∞ state and detects swarm collapse via debounced occupancy checks, re-seeding only after consecutive confirmations.
- **Errors.** Users see plain messages only. Logs are rotated, redacted (home paths, e-mails, tokens), stripped of source lines and locals, and stored in:
  - Windows: `%LOCALAPPDATA%\SwarmArt\logs`
  - Linux: `~/.local/state/swarmart/logs`
- **Presets and PNG metadata.** These are untrusted input: size capped, recursion-safe, schema-clamped, with decompression-bomb protection and atomic writes. No pickle and no eval.

### `Headless rendering` 

        python -m swarmart --headless out.png --model physarum --size 1920x1080 --steps 1500 --seed 7
        
        python -m swarmart --headless out.png --model flow --evolve 8

## Tests

       pip install -e .[test]
        pytest
