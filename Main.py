import numpy as np
#import pandas as pd
import datetime
import time
import string
from nltk.corpus import wordnet
import fileinput

def CE(QA, A):
    Q = ""
    #QA = "bread"
    #A = "colonist"
    not_include = "malevolent"
    i = 0
    x = ""
    i2 = 0
    i = 0
    j = 0
    z = 0

    syn = list()

    QAE = QA.lower()
    AE = A.lower()

    AI = AE.translate(str.maketrans('', '', string.punctuation))
    QAI = QAE.translate(str.maketrans('', '', string.punctuation))
    print(AI)
    QA1 = QAI.split()
    A1 = AI.split()

    KEYS = open("KEYWORD.txt", 'r')
    KEY1 = KEYS.read()

    KEYWORD = KEY1.split()
    # print(KEYWORD)

    # Keyword Parser

    AKEY = [t for t in A1 if t not in KEYWORD]
    QAKEY = [t for t in QA1 if t not in KEYWORD]

    # print(KeywordA)

    result = all(elem in A1 for elem in QA1)
    result1 = all(elem in QA1 for elem in A1)

    keyresult = all(elem in QAKEY for elem in AKEY)
    keyresult1 = all(elem in AKEY for elem in QAKEY)

    print(AKEY)
    print(QAKEY)

    if "not" in QA1 and "not" not in A1 or "Didn't" in QA1 and "Didn't" not in A1 or "not" in A1 and "not" not in QA1 or "Didn't" in A1 and "Didn't" not in QA1:
        x = 'Incorrect'

    elif result == True and result1 == True:
        print(result, "RES")
        x = "Correct"

    else:
        # all the keyword are equal
        if keyresult == True and keyresult1 == True:
            x = "Correct"
            print(keyresult, "KEY")
            print(keyresult1)

        else:
            # synonym parseing
            i3 = 0
            i2 = 0
            i = 0
            j = 0
            z = 0
            syn = list()

    # This is the slower
    # get all synonyms for every elem in list by iterating [+i]
    while i < len(AKEY):
        start_time = time.time()
        for synset in wordnet.synsets(AKEY[0 + i]):
            for lemma in synset.lemmas():
                syn.append(lemma.name())
        i += 1
        end_time = time.time()
        print("total time taken this loop: ", end_time - start_time)
    for i in AKEY:
        syn.append(i)
    if not_include in syn:
        syn.remove(not_include)
        print("removed:1")
    print("Hey" + str(syn))
    if not_include in syn:
        syn.remove(not_include)
        print("removed:1")

    # check if they match, try every synonym made with AKEY with every word in QAKEY

    while i2 < len(QAKEY):

        if len(QAKEY) != len(AKEY):
            x = "Incorrect"

            break
        elif len(syn) == 0:
            x = "Incorrect"
            break
        else:
            while z < len(syn):
                if syn[0 + z] == QAKEY[0 + j]:
                    print(syn[z])
                    print(QAKEY[j], QAKEY)
                    j += 1
                    i2 += 1
                    z = 0
                    break
                else:
                    z += 1

                if z == len(syn):
                    x = "Incorrect"
                    i2 = len(QAKEY) + 2
                    break

            if i2 < len(QAKEY):
                x = "Incorrect"

            if i2 == len(QAKEY):
                x = "Correct"
    print(x)
    return x


def know(statement):

    VER = open("verbs", 'r')

    VER1 = VER.read()

    verbs = VER1.split()

    #print(verbs)

    SUB = ""    #LOCATE SUBJECT

    ATRI = ""   #LOCATE ATRIBUTE

    SA = statement.lower()

    S1 = SA.split()

    KEYS = open("KEYWORD.txt", 'r')

    KEY1 = KEYS.read()

    KEYWORD = KEY1.split()
    # //////////////////////////////////
    POS = open("POS.txt", 'r')

    POS1 = POS.read()

    POSITIVE = POS1.split()
    # /////////////////////////////////

    NEG = open("NEG.txt", 'r')
    NEG1 = NEG.read()
    NEGATIVE = NEG1.split()

    # ////////////////////////////////
    # parse keywords

    SKEY = [t for t in S1 if t not in KEYWORD]
    SKEY1 = [k for k in SKEY if k not in NEGATIVE]
    SKEY2 = [l for l in SKEY1 if l not in POSITIVE]
    SKEY2 = [i for i in SKEY2 if i not in POS]
    SKEY_S = ""
    SKEY_S = SKEY_S.join(SKEY2)
    #print(SKEY_S)
    # Finish that up right now

    #print("test", SA)

    ii = 0
    ii2 = 0
    #print("LEN:", len(S1))

    while ii < len(S1)-1:
        while ii2 < len(verbs)-1:
            if S1[ii] == verbs[ii2]:
                print(S1[ii], verbs[ii2])
                SUB = S1[ii-1]
                print("SUB", S1[ii-1])
                ii = len(S1) + 2
                break
            else:
                ii2 += 1
        ii += 1
        ii2 = 0

    if SUB == "":
        print("ML TIME!")
        RL = 0

        SA = statement.lower()

        SA1 = SA.split()
        size = len(SA1)
        # print(size)

        VER = open("verbs", 'r')

        VER1 = VER.read()

        verbs = VER1.split()
        ii = 0
        ii2 = 0
        while ii < len(SA1) - 1:
            while ii2 < len(verbs) - 1:
                if SA1[ii] == verbs[ii2]:
                    # print(S1[ii], verbs[ii2])
                    SUB = SA1[ii - 1]
                    # print("SUB", S1[ii-1])
                    ii = len(SA1) + 2
                    break
                else:
                    ii2 += 1
            ii += 1
            ii2 = 0
        if SUB != "":
            SUBPOS = SA1.index(SUB)

            # BEGIN THE LEARNING

            actions = ["forward", "backward", "select"]
            incorrect_penalty = 5
            correct_reward = 10

            EPS = 0.9
            ALPHA = 0.1
            GAMMA = 0.9
            FRESH_TIME = 0.01
            EPISODES = 10

            def Qtable(SIZE, ACTIONS):
                table = pd.DataFrame(
                    np.zeros((size, len(actions))),  # q_table initial values
                    columns=actions,  # actions's name
                )
                # print(table)
                return table

            def Choose(state, q_table):
                state_actions = q_table.iloc[state, :]
                if (np.random.uniform() > EPS) or ((state_actions == 0).all()):  # explore by randomly selecting a move
                    action_name = np.random.choice(actions)
                else:  # exploit/ go to highest value
                    action_name = state_actions.idxmax()
                return action_name

            def EnvQuality(POS, ACT):
                if ACT == "forward":
                    R = 0

                    # if you hit a wall
                    if POS == size - 2:
                        POS_ = POS

                    # if you do not hit a wall
                    else:
                        POS_ = POS + 1

                elif ACT == "backward":
                    R = 0
                    # Hit a wall
                    if POS == 0:
                        POS_ = POS
                    # Do not
                    else:
                        POS_ = POS - 1

                else:
                    # If you select right

                    if POS == SUBPOS:
                        POS_ = 'correct'
                        R = 1
                        global RL
                        RL = SA1[POS]
                    else:
                        POS_ = 'incorrect'
                        R = 0

                return POS_, R

            def update_env(POS, episode, step_counter):
                # This is how environment be updated
                env_list = SA1
                if POS == 'correct':

                    interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter) #Useful FString
                    print('\r{}'.format(interaction), end='')
                    time.sleep(2)
                    print('\r                                ', end='')

                else:
                    # env_list[POS] = 'o'
                    # interaction = ''.join(env_list)
                    # print('\r{}'.format(interaction), end='')
                    # time.sleep(FRESH_TIME)
                    pass

            # Qtable(3, 3)

            def rl():

                Q = Qtable(size, actions)

                for episode in range(EPISODES):
                    step_counter = 0
                    POS = 0
                    is_terminated = False
                    # update_env(S, episode, step_counter)
                    while not is_terminated:

                        A = Choose(POS, Q)
                        POS_, R = EnvQuality(POS, A)  # take action & get next state and reward
                        q_predict = Q.loc[POS, A]
                        if POS_ != 'correct' and POS_ != 'incorrect':
                            q_target = R + GAMMA * Q.iloc[POS_, :].max()  # next state is not terminal
                        else:
                            q_target = R  # next state is terminal
                            is_terminated = True  # terminate this episode
                            episode = EPISODES + 1
                            break

                        Q.loc[POS, A] += ALPHA * (q_target - q_predict)  # update
                        POS = POS_  # move to next state

                        update_env(POS, episode, step_counter + 1)
                        step_counter += 1
                # RETURN = SA1[POS]
                # print(RETURN)
                return Q

            if __name__ == "__main__":
                q_table = rl()
                print('\r\nQ-table:\n')
                print(RL)
                SUB = RL
        else:
            SUB = SKEY_S

    iii = 0
    iii2 = 0
    while iii < len(S1)-1:
        while iii2 < len(verbs)-1:
            if S1[iii] == verbs[iii2]:
                #print(S1[iii], verbs[iii2])

                ATRI = S1[iii+1]
                #print("ATRI", S1[iii+1])
                iii = len(S1) + 2
                break
            else:
                iii2 += 1
        iii += 1
        iii2 = 0





    info = ""
    value = 0
    z = 0
    y = 0

    while y < len(SKEY):
        if SKEY[z] in NEGATIVE:
            value -= 1

        if SKEY[z] in POSITIVE:
            value += 1
        y += 1
        z += 1

    print(SKEY, value)
    timestamp = datetime.datetime.now()
    a = False
    a1 = False
    with open("KNOWLEDGE_FIELD.txt", "r") as KF:
            for line in KF:
                line = line.rstrip()  # remove '\n' at end of line
                if SA in line:
                    print(line)
                    a = True
                    break
    #Check if value is already in file, if yes do not write else write
    with open("KNOWLEDGE_FIELD.txt", "r") as KF1:
            for line in KF1:
                line = line.rstrip()  # remove '\n' at end of line
                if SKEY_S in line:
                    a1 = True
                    break
    line_num = 0
    #Find line with SKEY_S in it if it exists
    with open("KNOWLEDGE_FIELD.txt", "r") as KF1:
        for c, value1 in enumerate(KF1, 1):
            if SKEY_S in value1:
                #line = value.rstrip()
                a1 = True
                #print(line)
                break
            #line_num = c

    ATRI1 = "Attribute" + ":" + "" + ATRI
    if not a:
        with open("KNOWLEDGE_FIELD.txt", "a") as Know:
            if not a1:
                Know.write(f'\n{SA, SUB, ATRI1, value, timestamp}\n')
            #if a1:
                #Know.write(f'\n {SA, SKEY_S, value, timestamp}\n')

    mainvar = SA, SKEY_S, value, timestamp
    print("hey", mainvar)
    if not a and a1:
        f = open('KNOWLEDGE_FIELD.txt', 'r+')
        f.seek(0, c)
        f.write(f'     {SA, SUB, value, ATRI1, timestamp}')
        f.close()
    print("SUBJECT: ", SUB)
    print("ATRIBUTE: ", ATRI)


def search(statement):
   VER = open("verbs", 'r')

   VER1 = VER.read()
   verbs = VER1.split()

   SA = statement.lower()

   S1 = SA.split()

   KEYS = open("KEYWORD.txt", 'r')

   KEY1 = KEYS.read()

   KEYWORD = KEY1.split()

   POS = open("POS.txt", 'r')

   POS1 = POS.read()

   POSITIVE = POS1.split()

   SUB = ""

   NEG = open("NEG.txt", 'r')
   NEG1 = NEG.read()
   NEGATIVE = NEG1.split()


   # parse keywords
   ii = 0
   ii2 = 0
   while ii < len(S1) - 1:
       while ii2 < len(S1) - 1:
           if S1[ii] == verbs[ii2]:
               # print(S1[ii], verbs[ii2])
               SUB = S1[ii - 1]
               # print("SUB", S1[ii-1])
               ii = len(S1) + 2
               break
           else:
               ii2 += 1
       ii += 1
       ii2 = 0

   SKEY = [t for t in S1 if t not in KEYWORD]
   SKEY1 = [k for k in SKEY if k not in NEGATIVE]
   SKEY2 = [l for l in SKEY1 if l not in POSITIVE]
   SKEY_S = ""
   SKEY_S = SKEY_S.join(SKEY2)
   global line1
   with open("KNOWLEDGE_FIELD.txt") as search:
        for line in search:
            #print("I")
            line = line.rstrip()  # remove '\n' at end of line
            if SKEY_S in line:
                print(line)
                break
            else:
                line2 = "Sorry, I could not find the needed information to answer this question."


def verify(input):

   SV = input

   search(SV)



   #search variable = SV

   #CE(SV, MS)




   print(line1)

   #UPGRADE TO USE CE
   if SV in line1:
       x = "Correct"

   if x == 'Correct':
       print("yes")
   else:
       print("no")


def summarize(text):
    x = text
    x = 10


def answer(question, phrase):
    import numpy as np
    import pandas as pd
    import time
    VER = open("verbs", 'r')

    SUB = ""
    phrase = phrase.lower()
    ogphrase = phrase

    VER1 = VER.read()

    verbs = VER1.split()

    KEYWORD = open("KEYWORD.txt", 'r')

    KEY = KEYWORD.read()

    KEYWORD = KEY.split()

    question1 = question.lower()

    quest = question1.split()

    phrase1 = phrase.lower()
    passage = phrase.split(".")
    phrase = phrase1.split()
    for i in quest:
        if i == "the":
            quest.remove(i)
    ii = 0
    ii2 = 0
    while ii < len(quest) - 1:
        while ii2 < len(verbs) - 1:
            if quest[ii] == verbs[ii2]:
                # print(S1[ii], verbs[ii2])
                if ("how" in quest):
                    SUB = quest[ii + 1]
                elif ("what" in quest):
                        SUB = quest[ii - 1]
                #print("?", SUB)
                # print("SUB", S1[ii-1])
                ii = len(quest) + 2
                break
            else:
                ii2 += 1
        ii += 1
        ii2 = 0

    if SUB == "":
        RL = 0
        AI = question
        SA = question
        SUB = ""
        SA1 = SA.split()
        size = len(SA1)
        # print(size)

        VER = open("verbs", 'r')

        VER1 = VER.read()

        verbs = VER1.split()
        ii = 0
        ii2 = 0
        while ii < len(SA1) - 1:
            while ii2 < len(verbs) - 1:
                if SA1[ii] == verbs[ii2]:
                    # print(S1[ii], verbs[ii2])
                    SUB = SA1[ii - 1]
                    # print("SUB", S1[ii-1])
                    ii = len(SA1) + 2
                    break
                else:
                    ii2 += 1
            ii += 1
            ii2 = 0
        if SUB in SA1:
            SUBPOS = SA1.index(SUB)
            actions = ["forward", "backward", "select"]
            incorrect_penalty = 5
            correct_reward = 10

            EPS = 0.9
            ALPHA = 0.1
            GAMMA = 0.9
            TIME = 0.01
            EPISODES = 1

            def Qtable(SIZE, ACTIONS):
                table = pd.DataFrame(
                    np.zeros((size, len(actions))),  # q_table initial values
                    columns=actions,  # actions's name
                )
                return table

            def Choose(state, q_table):
                state_actions = q_table.iloc[state, :]
                if (np.random.uniform() > EPS) or ((state_actions == 0).all()):  # explore by randomly selecting a move
                    action_name = np.random.choice(actions)
                else:  # exploit/ go to highest value
                    action_name = state_actions.idxmax()
                return action_name

            def EnvQuality(POS, ACT):
                if ACT == "forward":
                    R = 0

                    # if you hit a wall
                    if POS == size - 2:
                        POS_ = POS

                    # if you do not hit a wall
                    else:
                        POS_ = POS + 1

                elif ACT == "backward":
                    R = 0
                    # Hit a wall
                    if POS == 0:
                        POS_ = POS
                    # Do not
                    else:
                        POS_ = POS - 1

                else:
                    # If you select right

                    if POS == SUBPOS:
                        POS_ = 'correct'
                        R = 1
                        global AI
                        AI = SA1[POS]
                    else:
                        POS_ = 'incorrect'
                        R = 0

                return POS_, R

            def update_env(POS, episode, step_counter):
                # This is how environment be updated
                env_list = SA1
                if POS == 'correct':

                    interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)  # Useful FString
                    print('\r{}'.format(interaction), end='')
                    time.sleep(2)
                    print('\r                                ', end='')

                else:
                    pass

            # Qtable(3, 3)

            def rl():
                # main part of RL loop
                q_table = Qtable(size, actions)
                for episode in range(EPISODES):
                    step_counter = 0
                    POS = 0
                    is_terminated = False
                    update_env(POS, episode, step_counter)
                    while not is_terminated:

                        A = Choose(POS, q_table)
                        POS_, R = EnvQuality(POS, A)  # take action & get next state and reward
                        q_predict = q_table.loc[POS, A]
                        if POS_ != 'correct' and POS_ != 'incorrect':
                            q_target = R + GAMMA * q_table.iloc[POS_, :].max()  # next state is not terminal
                        else:
                            q_target = R  # next state is terminal
                            is_terminated = True  # terminate this episode

                        q_table.loc[POS, A] += ALPHA * (q_target - q_predict)  # update
                        POS = POS_  # move to next state

                        update_env(POS, episode, step_counter + 1)
                        step_counter += 1
                return q_table

            if __name__ == "__main__":
                q_table = rl()
                #print('\r\nQ-table:\n')
                # print(q_table)
                #print(SA1[SUBPOS])
                Subject = SA1[SUBPOS]
                print("")
        else:
            print("Subject not found")

        #print(Subject)
        SUB = Subject


    key = ["who", "what", "where", "when", "how"]
    iii = 0
    iii2 = 0
    while iii < len(quest)-1:
        while iii2 < len(verbs)-1:
            if quest[iii] == verbs[iii2]:
                #print(S1[iii], verbs[iii2])

                ATRI = quest[iii+1]
                #print("ATRI", S1[iii+1])
                iii = len(quest) + 2
                break
            else:
                iii2 += 1
        iii += 1
        iii2 = 0


    #If there is a given phrase
    if (phrase != []):

        POS = open("positional.txt", 'r')
        POS1 = POS.read()
        POS = POS1.split()
        P = False
        #print(ATRI)
        if ("where" in quest):
            SUB = ATRI

        SUB1 = ATRI
        print(ATRI)
        #print(SUB)
        if("" in ogphrase):
            #Find the correct sentence
            for i in range(len(passage)):
                #print(SUB)
                if SUB in passage[i]:
                    I = i
                    break
                elif ATRI in passage[i] and "who" in quest:
                    I = i
                    break
                else:
                    I = "N/A"

            print(passage)
            PASS = passage[I]
            PASS = PASS.split()
            print(PASS)
            #If it is a location question.

            if ("where" in quest):
                 j = 0
                 jj = 0
                 while(j < len(PASS) - 1):
                    while( jj < len(POS) - 1):
                        #print(POS[jj])
                        #print(phrase[j])
                        if (POS[jj] == PASS[j]):
                            #print("Success")
                            P = True
                            ind_p = POS[jj]
                            j = len(phrase) + 2
                            break
                        else:
                            P = False
                            jj += 1
                    jj = 0
                    j += 1
                 if ( not P ):
                    print("This question is incorrectly phrased.")
                 if (P):
                            #This piece of code is a test of how the more simple code {In the below comment} works
                            for i in PASS:
                                if i in KEYWORD:
                                    PASS.remove(i)

                            #phrase2 = [i for i in phrase if i not in KEYWORD]

                            iii = 0
                            iii2 = 0
                            #Find location
                            while iii < len(PASS)-1:
                                while iii2 < len(POS)-1:
                                    if PASS[iii] == POS[iii2]:

                                        LOCATION = PASS[iii+1]

                                        iii = len(phrase) + 2
                                        break
                                    else:
                                        iii2 += 1
                                        #print(PASS[iii], POS[iii2])
                                iii += 1
                                iii2 = 0
                            SUB = ATRI
                            #To was removed from KEY
                            print(SUB, "is at the location:", LOCATION)


            elif("what" in quest or "how" in quest or "who" in quest):
                #print(PASS)


                #print(quest)
                ii = 0
                ii2 = 0
                while ii < len(PASS)-1:
                    while ii2 < len(verbs)-1:
                        if PASS[ii] == verbs[ii2]:
                            #print(PASS[ii], verbs[ii2])
                            RES = PASS[ii+1]
                            #print(RES)
                            VER = PASS[ii]
                            #print("SUB", S1[ii-1])
                            ii = len(PASS) + 2
                            break
                        else:
                            ii2 += 1
                    ii += 1
                    ii2 = 0
                ii = 0
                ii2 = 0
                while ii < len(PASS)-1:
                    while ii2 < len(verbs)-1:
                        if PASS[ii] == verbs[ii2]:
                            #print(PASS[ii], verbs[ii2])
                            RESNEW = PASS[ii-1]
                            #print(RES)
                            VER = PASS[ii]
                            #print("SUB", S1[ii-1])
                            ii = len(PASS) + 2
                            break
                        else:
                            ii2 += 1
                    ii += 1
                    ii2 = 0
                if ("how" in quest):
                    print(SUB1, VER, RES)
                elif("what" in quest and quest[1] in verbs):
                    print(SUB, VER, RES)
                elif("who" in quest):
                    print(RESNEW, VER, RES)
                else:
                    print("that specification is unavailable")




    else:
        with open("KNOWLEDGE_FIELD.txt") as search:
            for line in search:
                line = line.rstrip()  # remove '\n' at end of line
                if SUB in line:
                    #print(line)
                    a3 = True
                    line1 = line
                    break
                else:
                    line2 = "Sorry, I could not find the needed information to answer this question."

        #print(line1)
        key = ["who", "what", "where", "when", "how"]

        q1 = 0
        q = 0
        while q1 < len(quest):
            while q < len(key):
                if key[q] == quest[q1]:
                    a = True
                    q1 = len(quest) + 2
                    break
                else:
                    q += 1
                    a = False
            q1 += 1
            q = 0

        if (a):
            try:
                line1
            #IT FAILS
            except NameError:
                print(line2)
            #IT WORKS
            else:
                C = False
                result = line1.split(",")
                print(result)
                prop = result[2]
                prop = prop.split(":")
                prop = prop[1]
                prop = prop.translate(str.maketrans('', '', string.punctuation))
                #print(prop)
                if ("what" in quest or "What" in quest):
                    if ("color" in quest or "colour" in quest):
                        with open("colors.txt") as search:
                            for i in search:
                                i = i.rstrip()
                                if prop == i:
                                    #print (i)
                                    C = True
                                    break
                                else:
                                    C = False
                                    print("Unknown Property")
                                    break

                    elif (quest[1] not in verbs):
                        print("That specification is not available")
                    else:
                        if (quest[1] in verbs):
                            print(quest[1])
                            print(prop)

        else:
            print("You entered ",question)
            print("That is not a question")

CE("pie", "evil")