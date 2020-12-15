# -*- coding: utf-8 -*-

import sys

def progress_bar(current, total, note=''):
    length = 71
    completed = int(round(length * current / float(total)))

    dude = u"\U0001F40C"

    percentage = 100.0 * current / float(total)
    bar = '{}{}{}'.format('_' * completed , dude, '_' * (length - completed + len(dude)))

    sys.stdout.write('[{}] {}% ...{}{}'.format(bar[0:length+len(dude)], round(percentage,1), note, '\r' if percentage < 100.0 else '\n'))
    sys.stdout.flush()
