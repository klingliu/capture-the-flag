# Capture the Flag Competition AI Agents

These agents were written based on code from the UC Berkeley artificial intelligence course. All code was written in collaboration between Karen Liu, Class of 2023, and Tina Zhang, Class of 2024, and are intended to be in a 1v1 competition with other teams' agents, where the goal is to collect as many pellets as possible while avoiding the other team's ghosts.

The strategy adopted for our agents is to have one agent be offensive, collecting as many pellets as possible in enemy territory, while the other agent is defensive, guarding the home territory from enemy agents. The defensive agent is designed to switch modes to offensive when in proximity to a ghost. The concepts employed in our code include breadth first search and extensive particle filtering, where we analyze the probabilistic positions of the ghosts, the "noisy distances" being the approximate distances between the ghosts and our own agents.
