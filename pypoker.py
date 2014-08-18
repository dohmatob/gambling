#!/usr/bin/env python
from pokerengine.pokergame import PokerGameServer
from pokerengine.pokerchips import PokerChips
from random import choice
from getopt import getopt, GetoptError
from sys import argv, exit, stderr
from string import join
import gettext
gettext.install('textpoker', '.')

self_voicing = False
autoPlay = False

try:
    from speechd.client import SSIPClient
    speechd = SSIPClient("textpoker")
    def info(text):
        if self_voicing:
            speechd.speak(text)
        print text
except:
    def info(text):
        print text

players = 5
game = None
mySerial = 0

def serial2name(serial):
    global mySerial
    if serial == mySerial:
        return "you"
    else:
        return "Player %d" % serial

def setupGame(argv):
    """Process command line arguments and setup game"""

    def usage():
        print >>stderr, "Usage: %s [-a] [-n PLAYERS] [-s]" % argv[0]

    global autoPlay, players, game, mySerial, self_voicing
    try:
        opts, args = getopt(argv[1:], "an:s", ["help"])
    except GetoptError:
        usage()
        exit(2)
    for opt, arg in opts:
        if opt == "-a":
            autoPlay = True
        elif opt == "--help":
            usage()
            exit()
        elif opt == "-n":
            try:
                players = int(arg)
            except:
                print >>stderr, "%s is not a valid argument for %s" % (arg, opt)
                usage()
                exit(3)
        elif opt == "-s":
            self_voicing = True
    game = PokerGameServer("poker.%s.xml", ['/etc/poker-engine'])
    game.verbose = 0
    game.setVariant("holdem")
    game.setBettingStructure("10-15-pot-limit")
    serials = [n+1 for n in range(players)]
    if not autoPlay:
        mySerial = choice(serials)
    game.registerCallback(pokerEvent)
    for serial in serials:
        game.addPlayer(serial)
        game.payBuyIn(serial, 1500*100)
        game.sit(serial)
        if serial == mySerial and not autoPlay:
            game.autoBlindAnte(serial)
        else:
            game.botPlayer(serial)

def pokerEvent(id, type, *args):
    global game, mySerial
    if type == "all-in":
        if args[0] == mySerial:
            info(_("You are all in"))
        else:
            info(_("Player %d is all in") % args[0])
    elif type == "bet2pot":
        pass
    elif type == "blind":
        if args[0] == mySerial:
            info(_("You pay blind %s") % PokerChips.tostring(args[1]))
        else:
            info(_("Player %d pays blind %s") % (args[0], PokerChips.tostring(args[1])))
    elif type == "call":
        if args[0] == mySerial:
            info(_("You call %s") % PokerChips.tostring(args[1]))
        else:
            info(_("Player %d calls %s") % (args[0], PokerChips.tostring(args[1])))
    elif type == "check":
        if args[0] == mySerial:
            info(_("You check"))
        else:
            info(_("Player %d checks") % args[0])
    elif type == "end":
        (winners, showdown_stack) = args
        if showdown_stack:
            game_state = showdown_stack[0]
            if not game_state.has_key('serial2best'):
                serial = winners[0]
                if serial == mySerial:
                    info(_("You receive %(amount)s (everyone else folded)")
                         % { 'amount' : PokerChips.tostring(game_state['serial2share'][serial])})
                else:
                    info(_("Player %(number)d receives %(amount)s (everyone else folded)")
                         % { 'number': serial,
                             'amount': PokerChips.tostring(game_state['serial2share'][serial])})
            else:
                serial2displayed = {}
                hands = showdown_stack[0]['serial2best']
                for frame in showdown_stack[1:]:
                    message = None
                    if frame['type'] == 'left_over':
                        if frame['serial'] == mySerial:
                            message = _("You receive %(amount)s odd chips") % { 'amount' : PokerChips.tostring(frame['chips_left']) }
                        else:
                            message = _("Player %(number)s receives %(amount)s odd chips") % { 'number' :  frame['serial'], 'amount' : PokerChips.tostring(frame['chips_left']) }
                    elif frame['type'] == 'uncalled':
                        message = _("returning uncalled bet %(amount)s to %(name)s") % { 'amount' : PokerChips.tostring(frame['uncalled']), 'name' : serial2name(frame['serial']) }
                    elif frame['type'] == 'resolve':
                        best = { 'hi': 0,
                                 'low': 0x0FFFFFFF }
                        for serial in frame['serials']:
                            for side in ('hi', 'low'):
                                if not hands.has_key(serial):
                                    continue
                                hand = hands[serial]
                                if not hand.has_key(side):
                                    continue
                                if hand[side][1][0] == 'Nothing':
                                    continue

                                hand = hand[side]
                                show = False
                                if ( ( side == 'hi' and best['hi'] <= hand[0] ) or
                                     ( side == 'low' and best['low'] >= hand[0] ) ):
                                    best[side] = hand[0]
                                    show = True

                                if serial2displayed.has_key(serial) and not serial in frame[side]:
                                    # If the player already exposed the hand and is not going
                                    # to win this side of the pot, there is no need to issue
                                    # a message.
                                    continue

                                if show:
                                    serial2displayed[serial] = True
                                    value = game.readableHandValueLong(side, hand[1][0], hand[1][1:])
                                    if serial == mySerial:
                                        info(_("You show %(value)s") % { 'value' : value, 'side' : side })
                                    else:
                                        info(_("Player %(number)d shows %(value)s") % { 'number' : serial, 'value' : value, 'side' : side })
                                else:
                                    if serial == mySerial:
                                        info(_("You muck loosing hand"))
                                    else:
                                        info(_("Player %(number)d mucks loosing hand") % { 'number': serial })

                        for side in ('hi', 'low'):
                            if not frame.has_key(side):
                                continue
                            if len(frame[side]) > 1:
                                msg = join([ serial2name(serial) for serial in frame[side] ], ", ")
                                msg += _(" tie") % { 'side' : side }
                                info(msg)
                            else:
                                if frame[side][0] == mySerial:
                                    info(_("You win") % { 'side' : side })
                                else:
                                    info(_("Player %(number)d wins") % { 'number':frame[side][0], 'side': side })

                        if len(frame['serial2share']) > 1:
                            msg = _("winners share a pot of %(pot)s") % { 'pot' : PokerChips.tostring(frame['pot']) }
                            if frame.has_key('chips_left'):
                                msg += _(" (minus %(chips_left)d odd chips)") % { 'chips_left' : frame['chips_left'] }
                            info(msg)

                        for (serial, share) in frame['serial2share'].iteritems():
                            if serial == mySerial:
                                info(_("You receive %(amount)s") % { 'amount' : PokerChips.tostring(share) })
                            else:
                                info(_("Player %(number)d receives %(amount)s") % { 'number': serial, 'amount' : PokerChips.tostring(share) })

                    if message:
                        info(message)
        else:
            print "ERROR empty showdown_stack"
    elif type == "fold":
        if args[0] == mySerial:
            info(_("You fold"))
        else:
            info(_("Player %d folds") % args[0])
    elif type == "end_round_last":
        pass
    elif type == "finish":
        pass
    elif type == "game":
        (level, hand_serial, hands_count, time, variant, betting_structure, player_list, dealer, serial2chips) = args
        info(_("Hand #%(hand_serial)d, %(variant)s, %(betting_structure)s") % { 'hand_serial' : hand_serial, 'variant' : _(variant), 'betting_structure' : _(betting_structure) })
        if len(set(serial2chips.values())) == 1:
            info(_("All players have %s in chips") % PokerChips.tostring(serial2chips.values()[0]))
        else:
            for serial in player_list:
                if serial == mySerial:
                    info(_("You have %(chips)s in chips") % { 'chips' : PokerChips.tostring(serial2chips[serial]) })
                else:
                    info(_("Player %(serial)d has %(chips)s in chips") % { 'serial' : serial, 'chips': PokerChips.tostring(serial2chips[serial]) })
        if game.seats()[dealer] == mySerial:
            info(_("You receive the dealer button"))
        else:
            info(_("Player %d receives the dealer button") % game.seats()[dealer])
    elif type == "money2bet":
        pass
    elif type == "position":
        pass
    elif type == "raise":
        if args[0] == mySerial:
            info(_("You raise by %s") % PokerChips.tostring(args[1]))
        else:
            info(_("Player %d raises by %s") % (args[0], PokerChips.tostring(args[1])))
    elif type == "rake":
        pass
    elif type == "round":
        info(_(args[0]))
        if not autoPlay:
            info(_("Your cards: %s") % game.getHandAsString(mySerial))
        if not args[1].isEmpty():
            info(_("The board: %s") % game.getBoardAsString())
    elif type == "round_cap_decrease":
        pass
    elif type == "showdown":
        pass
    elif type == "sitOut":
        if args[0] == mySerial:
            info(_("Game over"))
            exit()
        else:
            info(_("Player %d is out of chips") % args[0])
    else:
        print "%s" % type
        print args


if __name__ == '__main__':
    setupGame(argv)
    if not autoPlay:
        info(_("You are player number %d on a table of %d players") % (mySerial, players))
    continuePlaying = True
    round = 0
    while continuePlaying and game.sitCount() > 1:
        round = round + 1
        game.beginTurn(round)
        if not autoPlay:
            while game.canAct(mySerial):
                action = None
                while (game.canAct(mySerial) and
                       action not in game.possibleActions(mySerial)):
                    action = raw_input(_("Your turn %s: ") % game.possibleActions(mySerial))
                    if action == "call":
                        game.call(mySerial)
                    elif action == "check":
                        game.check(mySerial)
                    elif action == "fold":
                        game.fold(mySerial)
                    elif action == "raise":
                        amount = input(_("How many chips? "))
                        game.callNraise(mySerial, amount*100)


     
