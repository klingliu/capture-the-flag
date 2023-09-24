# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

# All code in this file was written in collaboration between Karen Liu, Class of 2023, and Tina Zhang, Class of 2024.

selfParticles = [[], []]
selfNumParticles = 300


def createTeam(firstIndex, secondIndex, isRed,
               first='BFAgent', second='DefensiveReflexAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class MyNode:
    # state
    # most recent action
    # pointer to parent
    # total pathcost

    def __init__(self, state, parent, pathCost, action, GameState):
        self.state = state
        self.parent = parent
        self.pathCost = pathCost
        self.action = action
        self.GameState = GameState

    def getState(self):
        return self.state

    def getParent(self):
        return self.parent

    def getPathCost(self):
        return self.pathCost

    def getAction(self):
        return self.action

    def getGameState(self):
        return self.GameState


class ReflexCaptureAgent(CaptureAgent, object):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def __init__(self, index, timeForComputing=.1, scareTime=39, storedPath=[], x=0, Stuck=False,
                 posNF=[(0, 0), (0, 0), (0, 0), (0, 0)], n=0, listOfPath=util.Counter(), capsulePath=util.Counter(),
                 step=0):
        super(ReflexCaptureAgent, self).__init__(index, timeForComputing=.1)
        self.scareTime = scareTime
        self.storedPath = storedPath
        self.x = x
        self.Stuck = Stuck
        self.posNF = posNF
        self.n = n
        self.listOfPath = listOfPath
        self.capsulePath = capsulePath
        self.step = step

    def registerInitialState(self, gameState):
      self.start = gameState.getAgentPosition(self.index)
      CaptureAgent.registerInitialState(self, gameState)

      ## we want to store all the path to home from all legal positions
      self.initializeUniformlyParent(gameState, 0)
      # self.initializeUniformly(gameState, 1)

    def chooseActionOffensive(self, gameState):

      if self.step == 0:
        self.observeParent(gameState, 0)
        self.observeParent(gameState, 1)

        self.step = self.step + 1
      else:
        self.elapseTimeParent(gameState, 0)
        self.elapseTimeParent(gameState, 1)
        self.observeParent(gameState, 0)
        self.observeParent(gameState, 1)

      empty = util.Counter()
      list = (self.getBeliefDistributionParent(gameState, 0), self.getBeliefDistributionParent(gameState, 1))
      self.displayDistributionsOverPositions(list)

      agentState = gameState.getAgentState(self.index)
      conf = agentState.configuration
      ghostIsNear = False
      state = gameState.getAgentPosition(self.index)
      minGhostDist = self.checkGhost(gameState, state)
      Food = self.getFood(gameState)
      width = Food.packBits()[0]

      if (state[0] > (width / 2 - 1) and self.index % 2 != 1) or (state[0] < (width / 2 - 1) and self.index % 2 != 1):

        if minGhostDist <= 3:
          ghostIsNear = True

      action = self.breadthFirstSearch(gameState, self.getNearestFood(gameState), 2)
      headHome = self.timeToGoHome(gameState, 5)

      if headHome == True or ghostIsNear == True:
        action = self.breadthFirstSearch(gameState, (self.getHomePosition(gameState)), 4)

      elif self.scaredTimeBehavior(gameState) > 3:
        if ghostIsNear:
            ghostState = gameState.getAgentState(self.nearestGhostIndex(gameState))
            if ghostState.scaredTimer != 0:
                action = self.bf2(gameState,self.getNearestFood(gameState))
            else:
                action = self.breadthFirstSearch(gameState,self.getHomePosition(gameState),4)

        elif ghostIsNear == True and self.getCapsules(gameState) != []:
            action = self.breadthFirstSearch(gameState, self.getCapsulesPosition(gameState), 4)


        # self.storedPath = self.bf2(gameState, self.getNearestFood(gameState))
        #
        # if self.x < len(self.storedPath):
        #   # if action is legal!!!
        #   if self.storedPath[self.x] in gameState.getLegalActions(self.index):
        #     action = self.storedPath[self.x]
        #   self.x = self.x + 1
        # else:
        #   path = self.bf2(gameState, self.getNearestFood(gameState))
        #   if path != []:
        #
        #     action = self.bf2(gameState, self.getNearestFood(gameState))[0]
        #   else:
        #     action = []

      if self.n <= 3:
        self.posNF[self.n] = self.getNearestFood(gameState)
        self.n = self.n + 1
        if self.n == 3:
          # check
          if self.posNF[0] == self.posNF[2] and self.posNF[1] == self.posNF[3]:
            self.Stuck = True
          # reset

          self.n = 0

      if self.Stuck:
        path = self.bf2(gameState, self.getNearestFood(gameState))
        if path != []:

          action = self.bf2(gameState, self.getNearestFood(gameState))[0]
        else:
          action = []

      actions = gameState.getLegalActions(self.index)
      if action == []:
        return 'Stop'

      return action

    def scaredTimeBehavior(self, gameState):
        agentState = gameState.getAgentState(self.index)
        scaredTimeCountdown = agentState.scaredTimer
        return scaredTimeCountdown

    def nearestGhostIndex(self, gameState):
      # input a list of tuples (positions), returns a single tuple of the position, that is the position of the nearest ghost
      opponentIndex = self.getOpponents(gameState)  # list
      newGhostPositions = []
      for i in opponentIndex:
          newGhostPositions.append(gameState.getAgentPosition(i))

      selfPos = gameState.getAgentPosition(self.index)
      indicesAndPosition = util.Counter()
      for i in range(len(newGhostPositions)):
        indicesAndPosition[opponentIndex[i]] = util.manhattanDistance(selfPos, newGhostPositions[i])

      return min(indicesAndPosition, key=indicesAndPosition.get)

    def chooseActionDefensive(self, gameState):
      """
  Picks among the actions with the highest Q(s,a).
  """
      if self.step == 0:
        self.observeParent(gameState, 0)
        self.observeParent(gameState, 1)

        self.step = self.step + 1
      else:
        self.elapseTimeParent(gameState, 0)
        self.elapseTimeParent(gameState, 1)
        self.observeParent(gameState, 0)
        self.observeParent(gameState, 1)

      empty = util.Counter()

      list = (self.getBeliefDistributionParent(gameState, 0), self.getBeliefDistributionParent(gameState, 1))
      # if Use:
      self.displayDistributionsOverPositions(list)

      ghostPositions = self.getGhostPosition(gameState)
      selfPosition = gameState.getAgentPosition(self.index)

      posToChase = None

      # run particle filtering if ghost is on our side, but more than 5 squares away.
      # if it is on our side but less than 5 squares away, run bfs.
      # if it is not on our side, run offensive agent mode, or for now just go to the center // go home.
      whatAction = 0
      if ghostPositions == [None, None]:
        Agent1Inference = self.getBestBelief(gameState, self.getBeliefDistributionParent(gameState, 0))
        Agent2Inference = self.getBestBelief(gameState, self.getBeliefDistributionParent(gameState, 1))

        inferencePositions = [Agent1Inference, Agent2Inference]
        ghostsOnOurSide = []
        for i in inferencePositions:
          if self.onOurSide(gameState, i):
            ghostsOnOurSide.append(i)

        if ghostsOnOurSide == []:
          # offensive mode
          # but for now ...... go to home front line
          posToChase = self.getHomePosition(gameState)

        elif len(ghostsOnOurSide) == 2:
          posToChase = self.nearestGhost(gameState, ghostsOnOurSide)
        else:
          posToChase = ghostsOnOurSide[0]
          # len(ghostOnOurSide) ==1
          # the only thing there

      else:
        if ghostPositions[0] != None and ghostPositions[1] != None:
          posToChase = self.nearestGhost(gameState, ghostPositions)

        else:
            for i in ghostPositions:
                if i is not None and self.onOurSide(gameState,i):
                    posToChase = i

      if posToChase != None:
        actions = gameState.getLegalActions(self.index)
        path = self.bf2(gameState, posToChase)
        if path != []:
          action = path[0]
          whatAction = 1
        else:
          action = 'Stop'
          whatAction = 2

      else:

        actions = gameState.getLegalActions(self.index)

        action = random.choice(actions)
        whatAction = 3

      return action

    def initializeUniformlyParent(self, gameState, x):
      return self.initializeUniformly(gameState,x)

    def observeParent(self, gameState, x):
        return self.observe(gameState,x)

    def getBeliefDistributionParent(self, gameState, x):
      return self.getBeliefDistribution(gameState,x)

    def elapseTimeParent(self, gameState, x):
        return self.elapseTime(gameState,x)

    def getBestBelief(self, gameState, beliefs):
        # ourPos = gameState.getAgentPosition(self.index)
        # maxProbKey = None
        # maxProb = 0
        # minDistance = 999
        # for pos in beliefs:
        #   dist = util.manhattanDistance(ourPos, pos)
        #   if beliefs[pos] > maxProb:
        #     maxProb = beliefs[pos]
        #     if dist < minDistance:
        #       minDistance = dist
        #       maxProbKey = pos

        return max(beliefs, key=beliefs.get)

        # if there are several, tie break by the one closer to us

    def getPositionDistribution(self, gameState, position):
        # create a dictionary
        # check up down left right, if wall no
        # assign the key of the dictionary to be that legal posiution

        x, y = position
        # up down left right
        successors = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
        legalSuccessors = []
        for i in successors:
            a, b = i
            if not gameState.hasWall(a, b):
                legalSuccessors.append(i)

        # create a dictionary
        dist = util.Counter()
        for i in legalSuccessors:
            dist[i] = 1.0 / float(len(legalSuccessors))

        return dist

    def getHalfway(self, gameState):
        Food = self.getFood(gameState)
        width = Food.packBits()[0]
        height = Food.packBits()[1]

        halfway = (width / 2 - 1)

        if self.index % 2 == 1:
            halfway = halfway + 2

        return halfway

    def getLegalPositions(self, gameState):
        Food = self.getFood(gameState)
        legalPositions = []
        width = Food.packBits()[0]
        height = Food.packBits()[1]
        for i in range(width):
            for j in range(height):
                if not gameState.hasWall(i, j):
                    legalPositions.append((i, j))
        return legalPositions

    def bf2(self, currentGameState, OpponentPosition):
        """Search the shallowest nodes in the search tree first."""
        "*** YOUR CODE HERE ***"
        ## this is the actual bf search
        current = MyNode(currentGameState.getAgentPosition(self.index), None, 0, None, currentGameState)

        # Create the frontier (queue)
        frontier = util.Queue()
        frontier.push(current)

        # Create an empty set ("explored")
        explored = []
        while not frontier.isEmpty():
            # Pop from the frontier and assign that to node
            current = frontier.pop()
            # Check if the node.getState is a goal state, if so return # return plan (list)
            if OpponentPosition == current.getState():
                return self.getPlan(current)

            # Check if node.state is not in explored
            if current.getState() not in explored:
                # Add node to explored, add node's successors to frontier
                explored.append(current.getState())

                legalMoves = current.getGameState().getLegalActions(self.index)

                listOfChildren = []
                for action in legalMoves:
                    successorGameState = current.getGameState().generateSuccessor(self.index, action)
                    state = successorGameState.getAgentPosition(self.index)
                    actionUsed = action
                    CurrentPathCost = 1

                    listOfChildren.append((state, action, successorGameState, CurrentPathCost))

                for i in listOfChildren:
                    child = MyNode(i[0], current, current.getPathCost() + i[3], i[1], i[2])
                    parent = child.getParent()
                    frontier.push(child)

        # Return failure
        return []

    def getPlan(self, current):
        # Pop from the frontier and assign that to node
        # Check if the node.getState is a goal state, if so return # return plan (list)
        plan = []
        while current.parent is not None:
            plan.append(current.getAction())
            current = current.getParent()

        plan.reverse()
        return plan

    def getGhostPosition(self, gameState):
        opponentOIndex = self.getOpponents(gameState)  # list
        newGhostPositions = []
        for i in opponentOIndex:
            newGhostPositions.append(gameState.getAgentPosition(i))
        return newGhostPositions

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}

    def getHalfLegalPositionsParent(self, gameState):
        return self.getHalfLegalPositions(gameState)

    def timeToGoHome(self, currentGameState, carrylimit):
        foodLeft = len(self.getFood(currentGameState).asList())
        carryNumber = carrylimit

        # hardcoded change later
        if foodLeft <= 10 and self.scareTime == 0:
            carryNumber = 3

        foodEaten = currentGameState.getAgentState(self.index).numCarrying
        if foodEaten >= carryNumber:
            return True

        if foodLeft == 2:
            return True
        return False

    def getHomePosition(self, currentGameState):
        Pos = currentGameState.getAgentPosition(self.index)
        Food = self.getFood(currentGameState)
        width = Food.packBits()[0]
        height = Food.packBits()[1]

        halfway = (width / 2 - 1)

        if self.index % 2 == 1:
            halfway = halfway + 2
        homebaseList = []
        for i in range(0, height):
            homepos = (halfway, i)
            if not currentGameState.hasWall(homepos[0], homepos[1]):
                homebaseList.append(homepos)

        mindistance = 9999
        HomePosition = None

        for i in homebaseList:

            Dist = util.manhattanDistance(i, Pos)
            if Dist < mindistance:
                mindistance = Dist
                HomePosition = i

        return HomePosition

    def getNearestFood(self, currentGameState):

        Pos = currentGameState.getAgentPosition(self.index)
        Food = self.getFood(currentGameState)
        width = Food.packBits()[0]
        height = Food.packBits()[1]

        minFoodDist = 99999
        NearestFood = None

        # calculating food distance (want minimum / or the maximum reciprocal)
        for i in range(width):
            for j in range(height):
                if Food[i][j]:
                    foodDist = util.manhattanDistance((i, j), Pos)
                    if foodDist < minFoodDist:
                        minFoodDist = foodDist
                        NearestFood = (i, j)

        return NearestFood

    def getCapsulesPosition(self, gameState):
        listofCapsules = self.getCapsules(gameState)
        minDist = 99999
        location = None

        for i in listofCapsules:
            if i != None:
                capsuleLocation = i
                dist = util.manhattanDistance(capsuleLocation, gameState.getAgentPosition(self.index))

                if dist < minDist:
                    minDist = dist
                    location = capsuleLocation

        if location == None:
            return self.getHomePosition(gameState)
        return location

    def checkGhost(self, successorGameState, state):
        opponentOIndex = self.getOpponents(successorGameState)  # list
        minGhostDist = 99999
        newGhostPositions = []
        for i in opponentOIndex:
            newGhostPositions.append(successorGameState.getAgentPosition(i))

        for i in newGhostPositions:
            if i != None:
                newGhostLocation = i
                ghostDist = util.manhattanDistance(newGhostLocation, state)
                if ghostDist < minGhostDist:
                    minGhostDist = ghostDist

        return minGhostDist

    def breadthFirstSearch(self, currentGameState, foodLocation, level):
        """Search the shallowest nodes in the search tree first."""
        "*** YOUR CODE HERE ***"
        current = MyNode(currentGameState.getAgentPosition(self.index), None, 0, None, currentGameState)

        # Create the frontier (queue)
        # breadthfirst changed to USC
        # frontier = util.Queue()

        frontier = util.PriorityQueue()
        frontier.push(current, current.getPathCost())

        # Create an empty set ("explored")
        explored = []

        # Start loop (if frontier is not empty)
        while not frontier.isEmpty():

            # Pop from the frontier and assign that to node
            current = frontier.pop()

            # Check if the node.getState is a goal state, if so return # return plan (list)
            if foodLocation == current.getState():

                return self.getPlan(current)[0]

            # Check if node.state is not in explored
            if current.getState() not in explored:
                # Add node to explored, add node's sucessors to frontier
                explored.append(current.getState())

                ## while loop to loop through touples and create children
                # legalMoves = currentGameState.getLegalActions()
                cg = current.getGameState

                legalMoves = current.getGameState().getLegalActions(self.index)
                listOfChildren = []
                for action in legalMoves:
                    successorGameState = current.getGameState().generateSuccessor(self.index, action)
                    state = successorGameState.getAgentPosition(self.index)
                    actionUsed = action
                    CurrentPathCost = 1

                    # BreadthFirst Changed to USC
                    # if ghost is nearby, chose the one with lower path cost

                    # get ghost position

                    # opponentOIndex = self.getOpponents(successorGameState)  # list
                    # minGhostDist = 99999
                    # newGhostPositions = []
                    # for i in opponentOIndex:
                    #   newGhostPositions.append(successorGameState.getAgentPosition(i))
                    #
                    # for i in newGhostPositions:
                    #   if i != None:
                    #     newGhostLocation = i
                    #     ghostDist = util.manhattanDistance(newGhostLocation, state)
                    #     if ghostDist < minGhostDist:
                    #       minGhostDist = ghostDist
                    #
                    # Food = self.getFood(currentGameState)
                    # width = Food.packBits()[0]
                    # if state[0] > (width/2-1):
                    #
                    #   if minGhostDist <= level:
                    #     CurrentPathCost = (5-minGhostDist)*100

                    if self.scareTime <= 3:

                        minGhostDist = self.checkGhost(successorGameState, state)
                        Food = self.getFood(currentGameState)
                        width = Food.packBits()[0]
                        if (state[0] > (width / 2 - 1) and self.index % 2 != 1) or (
                                state[0] < (width / 2 - 1) and self.index % 2 != 1):

                            if minGhostDist <= level:
                                CurrentPathCost = (5 - minGhostDist) * 1000


                    listOfChildren.append((state, action, successorGameState, CurrentPathCost))

                for i in listOfChildren:
                    child = MyNode(i[0], current, current.getPathCost() + i[3], i[1], i[2])
                    parent = child.getParent()
                    frontier.push(child, child.getPathCost())

                    ## for loop through list
                    ## for x in list
                    ## child = myNode(,x,x)
                    ## state from tuple (at index 0)
                    # parent is current
                    # pathcost is current.getPathCost + cost from tuple (index 2)
                    # action from tuple (index 1)
                    # push child onto frontier

        # Return failure
        return []

    def nearestGhost(self, gameState, ghostPositions):
      # input a list of tuples (positions), returns a single tuple of the position, that is the position of the nearest ghost
      selfPos = gameState.getAgentPosition(self.index)
      listofPositions = util.Counter()
      for i in ghostPositions:
        listofPositions[i] = util.manhattanDistance(selfPos, i)
      return min(listofPositions, key=listofPositions.get)

    def onOurSide(self, gameState, pos):
      onOurSide = False
      if self.index % 2 == 0:
        # red team
        if pos[0] <= self.getHalfway(gameState):
          onOurSide = True
      else:
        if pos[0] >= self.getHalfway(gameState):
          onOurSide = True
      return onOurSide


class DefensiveReflexAgent(ReflexCaptureAgent, object):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

    def __init__(self, index, timeForComputing=.1, scareTime=39, storedPath=[], x=0, Stuck=False,
                 posNF=[(0, 0), (0, 0), (0, 0), (0, 0)], n=0, listOfPath=util.Counter(), capsulePath=util.Counter(),
                 step=0):
        super(DefensiveReflexAgent, self).__init__(index, timeForComputing=.1)
        self.scareTime = scareTime
        self.storedPath = storedPath
        self.x = x
        self.Stuck = Stuck
        self.posNF = posNF
        self.n = n
        self.listOfPath = listOfPath
        self.capsulePath = capsulePath
        self.step = step

    def chooseAction(self, gameState):
      return self.chooseActionDefensive(gameState)

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        ## we want to store all the path to home from all legal positions
        self.initializeUniformly(gameState, 0)
        self.initializeUniformly(gameState, 1)

    def getHalfLegalPositions(self, gameState):
        halfway = self.getHalfway(gameState)
        legalPositions = self.getLegalPositions(gameState)
        toReturn = []
        for pos in legalPositions:
            if pos[0] < halfway:
                toReturn.append(pos)

        return toReturn

    def initializeUniformly(self, gameState, x):
      global selfNumParticles
      global selfParticles
      for i in range(selfNumParticles):
        legalPositions = self.getHalfLegalPositions(gameState)
        legalPosIndex = i % len(legalPositions)
        toAppend = legalPositions[legalPosIndex]
        selfParticles[x].append(toAppend)

    def observe(self, gameState, x):
      noisyDistances = gameState.getAgentDistances()
      opponentIndexes = self.getOpponents(gameState)
      noisyDistanceOfOpponent = noisyDistances[opponentIndexes[x]]

      # noisyDistance = gameState.getAgentDistances()
      # noisyDistanceOfOpponent = []
      # opponentIndex = self.getOpponents(gameState)
      # for i in range(len(opponentIndex)):
      #   index = opponentIndex[i]
      #   noisyDistanceOfOpponent.append(noisyDistance[index])

      # for i in range(len(opponentIndex)):
      #   index = opponentIndex[i]
      #   noisyDistancesOfOpponent.append(noisyDistances[index])

      Use = True

      pacmanPosition = gameState.getAgentPosition(self.index)

      particleCounter = util.Counter()
      for p in selfParticles[x]:
        trueDistance = util.manhattanDistance(p, pacmanPosition)
        particleCounter[p] = particleCounter[p] + gameState.getDistanceProb(trueDistance, noisyDistanceOfOpponent)
      particleCounter.normalize()

      if particleCounter.totalCount() == 0:
        Use = False
        self.initializeUniformly(gameState, x)
      else:
        toSet = []
        for i in range(selfNumParticles):
          sample = util.sample(particleCounter)
          toSet.append(sample)
        selfParticles[x] = toSet
      return Use

    def elapseTime(self, gameState, x):
      # creates new particles
      # for every particle in self.particle, it gets the position distribution
      # sample from that
      new = []
      for oldParticle in selfParticles[x]:
        newPosDist = self.getPositionDistribution(gameState, oldParticle)
        sample = util.sample(newPosDist)
        new.append(sample)
      selfParticles[x] = new

    def getBeliefDistribution(self, gameState, x):
      beliefs = util.Counter()
      legalPositions = self.getLegalPositions(gameState)
      allPositions = legalPositions
      for p in allPositions:
        numP = 0
        for particle in selfParticles[x]:
          if particle == p:
            numP += 1
        beliefs[p] = float(numP) / float(selfNumParticles)

      max_key = max(beliefs, key=beliefs.get)

      for i in beliefs:
        if i != max_key:
          beliefs[i] = 0
        if i == max_key:
          beliefs[i] = 1

      return beliefs


class BFAgent(ReflexCaptureAgent, object):
    def __init__(self, index, timeForComputing=.1, scareTime=39, storedPath=[], x=0, Stuck=False, \
                 posNF=[(0, 0), (0, 0), (0, 0), (0, 0)], n=0, listOfPath=util.Counter(), capsulePath=util.Counter(),
                 step=0):
        super(BFAgent, self).__init__(index, timeForComputing=.1)
        self.scareTime = scareTime
        self.storedPath = storedPath
        self.x = x
        self.Stuck = Stuck
        self.posNF = posNF
        self.n = n
        self.listOfPath = listOfPath
        self.capsulePath = capsulePath
        self.step = step

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.initializeUniformly(gameState, 0)
        self.initializeUniformly(gameState, 1)

    def chooseAction(self, gameState):
      return self.chooseActionOffensive(gameState)

    def getHalfLegalPositions(self, gameState):
        halfway = self.getHalfway(gameState)
        legalPositions = self.getLegalPositions(gameState)
        toReturn = []
        for pos in legalPositions:
            if pos[0] > halfway:
                toReturn.append(pos)
        return toReturn

    def initializeUniformly(self, gameState, x):
      global selfNumParticles
      global selfParticles
      for i in range(selfNumParticles):
        legalPositions = self.getHalfLegalPositions(gameState)
        legalPosIndex = i % len(legalPositions)
        toAppend = legalPositions[legalPosIndex]
        selfParticles[x].append(toAppend)

    def observe(self, gameState, x):
      noisyDistances = gameState.getAgentDistances()
      opponentIndexes = self.getOpponents(gameState)
      noisyDistanceOfOpponent = noisyDistances[opponentIndexes[x]]

      # noisyDistance = gameState.getAgentDistances()
      # noisyDistanceOfOpponent = []
      # opponentIndex = self.getOpponents(gameState)
      # for i in range(len(opponentIndex)):
      #   index = opponentIndex[i]
      #   noisyDistanceOfOpponent.append(noisyDistance[index])

      # for i in range(len(opponentIndex)):
      #   index = opponentIndex[i]
      #   noisyDistancesOfOpponent.append(noisyDistances[index])

      pacmanPosition = gameState.getAgentPosition(self.index)

      particleCounter = util.Counter()
      for p in selfParticles[x]:
        trueDistance = util.manhattanDistance(p, pacmanPosition)
        particleCounter[p] = particleCounter[p] + gameState.getDistanceProb(trueDistance, noisyDistanceOfOpponent)
      particleCounter.normalize()

      if particleCounter.totalCount() == 0:
        self.initializeUniformly(gameState, x)
      else:
        toSet = []
        for i in range(selfNumParticles):
          sample = util.sample(particleCounter)
          toSet.append(sample)
        selfParticles[x] = toSet

    def getBeliefDistribution(self, gameState, x):
      beliefs = util.Counter()
      legalPositions = self.getHalfLegalPositions(gameState)
      allPositions = legalPositions
      for p in allPositions:
        numP = 0
        for particle in selfParticles[x]:
          if particle == p:
            numP += 1
        beliefs[p] = float(numP) / float(selfNumParticles)

      max_key = max(beliefs, key=beliefs.get)
      for i in beliefs:
        if i != max_key:
          beliefs[i] = 0
        if i == max_key:
          beliefs[i] = 1

      # for i in beliefs:
      #   if i!= max_key or beliefs[max_key]< 0.5:
      #     beliefs[i] = 0
      #   else:
      #     beliefs[i] = 1

      return beliefs

    def elapseTime(self, gameState, x):
      # creates new parciles
      # for every particle in self.particle, it gets the position distribution
      # sqmple from that
      new = []
      for oldParticle in selfParticles[x]:
        newPosDist = self.getPositionDistribution(gameState, oldParticle)
        sample = util.sample(newPosDist)
        new.append(sample)
      selfParticles[x] = new


