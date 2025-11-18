#!/usr/bin/env python3

DOWNLOADS = False
TOP_N = 5
REPLACE_WORD = "Moose"

CORPUS_DIR = './data/'

QUEUE_WORK = 'work'
QUEUE_FEEDBACK = 'feedback'

# Usage:
#
# ./main.py main N - starts the main process
# N is how many workers are expected
#
# ./main.py worker N - starts N workers
#

from typeguard import typechecked, install_import_hook, check_type

install_import_hook()

# Typeguard
import pika
import pathlib
import os
import nltk.sentiment
import nltk
import multiprocessing
import sys
import json
import shutil
from collections import Counter

###
#
# General section
#
###


###
#
# Main section
#
###

WORKERS = None


###
#
# Worker section
#
###

PROCESS_COUNT = None


@typechecked
def on_message(tid: int, channel, method_frame, header_frame, body):
    msg = json.loads(body)
    print(f'Worker {tid}: {msg}')
    data = None
    id = msg['id']
    filepath = CORPUS_DIR + msg['filepath']
    with open(filepath, errors='ignore') as f:
        first = msg['byte_first']
        last = msg['byte_last']

        f.seek(first)
        data = f.read(1024)
        try:
            if not data[0].isspace():
                first += len(data.split()[0])
        except Exception:
            pass

        f.seek(last)
        data = f.read(1024)
        try:
            if not data[0].isspace():
                last += len(data.split()[0])
        except Exception:
            pass

        f.seek(first)
        data = f.read(last - first)

    counts = Counter([x.lower() for x in data.split() if x.isalnum()])
    total = counts.total()
    top_n = counts.most_common(TOP_N * 5)

    sentiment = nltk.sentiment.SentimentIntensityAnalyzer().polarity_scores(data)

    tokens = nltk.word_tokenize(data)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    ne = set()
    for entity in entities:
        if hasattr(entity, 'label') and entity.label() in ('PERSON', 'ORGANIZATION', 'GPE'):
            name = ''.join(c[0] for c in entity)
            ne.add(name)

    ne = list(ne)
    ne.sort(key=len, reverse=True)
    for entity in ne:
        data = data.replace(entity, REPLACE_WORD)
    pathlib.Path(filepath + f'.{id}').write_text(data)

    data = data.replace('\n', '')
    data = data.replace('\r', '')
    data = data.split('.')
    data = [x.rstrip() for x in data if x.rstrip()]
    with open(filepath + f'.sentence_{id}', 'w') as f_out:
        for i in data:
            f_out.write(i)
            f_out.write('\n')

    channel.queue_declare(QUEUE_FEEDBACK)
    channel.basic_publish(exchange='',
                          routing_key=QUEUE_FEEDBACK,
                          body=json.dumps({
                              'total': total,
                              'top_n': top_n,
                              'sentiment': sentiment
                          }).encode('utf-8'))
    print(f'Worker {tid} done')
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)


@typechecked
def worker_fn(tid: int):
    print(f'Worker {tid} starting')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('172.17.0.2'))
    channel = connection.channel()

    channel.queue_declare(QUEUE_WORK)
    channel.basic_consume(QUEUE_WORK, lambda *args: on_message(tid, *args))
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    connection.close()

###
#
# CLI section
#
###


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[1] == 'main':
    WORKERS = int(sys.argv[2])
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('172.17.0.2'))
    channel = connection.channel()

    channel.queue_declare(QUEUE_WORK)

    for file in os.scandir(CORPUS_DIR):
        if not file.is_file(follow_symlinks=True):
            continue
        if not file.name.endswith('txt'):
            continue
        size = file.stat(follow_symlinks=True).st_size
        worksz = int(size / WORKERS)

        for id in range(WORKERS):
            print(f'Dispatching task {id}')
            channel.basic_publish(exchange='',
                                  routing_key=QUEUE_WORK,
                                  body=json.dumps({
                                      'id': id,
                                      'filepath': file.name,
                                      'byte_first': id * worksz,
                                      'byte_last': (id + 1) * worksz
                                  }).encode('utf-8'))

        total = 0
        top_n = {}
        sentiment = {'neg': 0, 'pos': 0, 'neu': 0}

        for id in range(WORKERS):
            print(f'Collecting task {id}')
            channel.queue_declare(QUEUE_FEEDBACK)
            while True:
                method_frame, header_frame, body = channel.basic_get(QUEUE_FEEDBACK)
                if body is None:
                    continue
                body = json.loads(body)
                total += body['total']
                for el in body['top_n']:
                    old = top_n.setdefault(el[0], 0)
                    top_n[el[0]] = old + el[1]
                sentiment['neg'] += body['sentiment']['neg']
                sentiment['pos'] += body['sentiment']['pos']
                sentiment['neu'] += body['sentiment']['neu']
                channel.basic_ack(method_frame.delivery_tag)
                break

        top_n = [(k, v) for k, v in sorted(top_n.items(), reverse=True, key=lambda item: item[1])]
        top_n = top_n[:TOP_N]
        sentiment['neg'] /= WORKERS
        sentiment['pos'] /= WORKERS
        sentiment['neu'] /= WORKERS

        with open(CORPUS_DIR + file.name + '.out', 'wb') as f_out:
            for id in range(WORKERS):
                path = f'{CORPUS_DIR}{file.name}.{id}'
                with open(path, 'rb') as fd:
                    shutil.copyfileobj(fd, f_out)
                os.remove(path)

        sentences = []
        for id in range(WORKERS):
            path = f'{CORPUS_DIR}{file.name}.sentence_{id}'
            with open(path) as fd:
                for line in fd:
                    sentences.append(line.rstrip())
            os.remove(path)

        sentences.sort(key=len, reverse=True)
        with open(CORPUS_DIR + file.name + '.sentences', 'w') as f_out:
            for i in sentences:
                f_out.write(i)
                f_out.write('\n')

        report = ""
        report += f'Total: {total}\n'
        report += f'Top N: {top_n}\n'
        report += f'Sentiment: {sentiment}\n'
        pathlib.Path("report.txt").write_text(report)


elif __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[1] == 'worker':
    PROCESS_COUNT = int(sys.argv[2])
    multiprocessing.set_start_method('spawn')

    if DOWNLOADS:
        # sentiment
        nltk.download('vader_lexicon')
        # ner
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('maxent_ne_chunker_tab')

    workers = [multiprocessing.Process(target=worker_fn, args=(id,))
               for id in range(PROCESS_COUNT)]
    [x.start() for x in workers]
    [x.join() for x in workers]
