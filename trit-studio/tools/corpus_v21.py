"""Reviewed deterministic context exercises. Templates are labelled, not independent stories.
Training and evaluation use different prompt templates. No model/user data/external calls.
Evaluation is public development evidence, not a hidden generalization benchmark.
"""
import hashlib,json

def row(parts, family, split='train', template=True):
    r={'messages':[{'role':'user' if i%2==0 else 'assistant','content':s} for i,s in enumerate(parts)],
       'family':family,'split':split,'origin':'authored-template-v21' if template else 'authored-synthetic-v21'}
    r['id']='ru-v21-'+hashlib.sha256(json.dumps(r['messages'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:20]
    return r

def build():
    train=[];challenge=[]
    names=['Лена','Нина','Вера','Оля','Рома','Паша','Илья','Миша']
    colors=['синяя','жёлтая','красная','белая','зелёная','чёрная']
    places=['на полке','в ящике','на столе','в сумке','у окна','под книгой']
    # Same final request, different facts. Correct answers must change with history.
    for i,name in enumerate(names):
        other=names[(i+3)%len(names)]
        for form in range(4):
            intro=[f'Меня зовут {name}.',f'В чате я {name}.',f'Зови меня {name}.',f'Моё имя {name}.'][form]
            ask=['Как меня зовут? Ответь одним именем.','Назови только моё имя.','Какое имя я назвал? Только имя.','Кто я? Напиши лишь имя.'][form]
            for reply in ['Понял.','Хорошо, запомню на время беседы.','Рад знакомству.']:
                train.append(row([intro,reply,ask,name+'.'],'v21_name'))
            train.append(row([intro,'Понял.',f'А моего друга зовут {other}.','Хорошо.',ask,name+'.'],'v21_people'))
            train.append(row([f'Сначала зови меня {other}.','Хорошо.',f'Теперь используй имя {name}.','Принято.',ask,name+'.'],'v21_rename'))
    for noun in ['папка','чашка']:
        for i,old in enumerate(colors):
            for new in colors:
                if old==new:continue
                for form in range(2):
                    intro=f'Моя {noun} {old}.' if form==0 else f'Запомни: {noun} {old}.'
                    change=f'Поправка: {noun} {new}.' if form==0 else f'Нет, я ошибся: {noun} {new}.'
                    ask=f'Какая {noun} теперь? Только цвет.' if form==0 else f'Назови актуальный цвет: какая {noun}? Одним словом.'
                    train.append(row([intro,'Понял.',change,'Принял поправку.',ask,new+'.'],'v21_color_change'))
    for noun in ['блокнот','билет']:
        for old in places:
            for new in places:
                if old==new:continue
                train.append(row([f'{noun.capitalize()} был {old}.','Понял.',f'Теперь {noun} лежит {new}.','Учту новое место.',f'Где {noun} сейчас? Ответь кратко.',new.capitalize()+'.'],'v21_location_change'))
    for fact,opposite,question in [
        ('Письмо отправлено.','Письмо пока не отправлено.','Письмо уже отправлено? Ответь да или нет.'),
        ('Встреча подтверждена.','Встреча ещё не подтверждена.','Встреча уже подтверждена? Ответь да или нет.'),
        ('Я купил билет.','Я ещё не купил билет.','Билет уже куплен? Ответь да или нет.'),
        ('Я закончил рисунок.','Я ещё не закончил рисунок.','Рисунок уже закончен? Ответь да или нет.')]:
        for prefix in ['Статус: ','На сейчас: ','Уточню: ']:
            for done in [True,False]:
                train.append(row([prefix+(fact if done else opposite),'Понял.',question,'Да.' if done else 'Нет.'],'v21_action'))
    # Unknown facts: do not guess a new personal detail just because a name/color appeared nearby.
    for name in names:
        for known,question in [('Мне нравится читать.','Какой у меня любимый цвет?'),('У меня есть блокнот.','Где сейчас мой блокнот?'),('Завтра я занят.','В какое время я освобожусь?')]:
            train.append(row([f'Я {name}. {known}','Понял.',question,'Ты пока не сообщил об этом.'],'v21_unknown'))
    # Natural short language variants, not new diagnostic answers hard-coded into inference.
    pairs=[
      ('Привет, есть минутка поговорить?','Привет! Да, я здесь. Как проходит день?'),
      ('Можно к тебе с обычной болтовнёй?','Конечно. О чём хочется поговорить?'),
      ('Я просто заглянул поздороваться.','Привет! Рад твоему сообщению.'),
      ('День был тихий, особо рассказывать нечего.','Иногда тихий день сам по себе хорош. Удалось отдохнуть?'),
      ('На улице сегодня приятно, я немного прошёлся.','Здорово. Ты гулял по знакомому маршруту?'),
      ('Сижу с кружкой чая и ничего не делаю.','Звучит как спокойная пауза. Можно никуда не торопиться.'),
      ('Впервые за неделю появилось свободное время.','Что сейчас больше хочется: отдохнуть или заняться чем-то приятным?'),
      ('Прочитал последнюю страницу, даже грустно расставаться с книгой.','Понимаю. К какому герою ты успел привязаться?'),
      ('Хочу показать рисунок, но сначала расскажу идею.','Расскажи. Что ты хотел передать рисунком?'),
      ('Не надо пока советов, я просто рассказываю.','Хорошо, слушаю. Не буду торопиться с советами.'),
      ('Я ошибся в прошлой фразе, можно поправить?','Конечно. Напиши верный вариант.'),
      ('Давай оставим эту тему и поговорим о выходных.','Давай. Ты уже что-нибудь запланировал?'),
      ('Не задавай сейчас вопросов, просто поддержи.','Понимаю. Иногда важно просто выдохнуть и побыть без лишних требований.'),
      ('Планы поменялись: останусь дома, а не пойду гулять.','Понял, прогулка отменяется. Сейчас план остаться дома.'),
      ('Ты спрашивал про фильм, я ещё его не досмотрел.','Понял. Тогда не будем обсуждать концовку.'),
      ('Я не просил решать проблему за меня.','Верно. Давай остановимся на том, что тебе сейчас нужно.'),
      ('Сделай мягче: «Ты всё забыл».','Кажется, это осталось без внимания. Напомню ещё раз.'),
      ('Сократи: «В настоящий момент я не могу ответить».','Сейчас не могу ответить.'),
      ('Скажи проще: «Следует отложить принятие решения».','Лучше решить это позже.'),
      ('Я собираюсь позвонить, но ещё не звонил.','Понял: звонок пока только в планах.'),
      ('Утром хотел кофе, теперь хочу чай.','Теперь ты хочешь чай.'),
      ('Я рассказал про брата, а не про себя.','Понял, это относится к брату. Не буду переносить это на тебя.'),
      ('Ответь одним предложением: зачем нужна закладка?','Закладка помогает найти место, на котором ты остановился.'),
      ('Что можно сказать человеку, который закончил трудное дело?','Здорово, что ты справился. Теперь можно немного отдохнуть.')]
    natural=[row([q,a],f'v21_natural_{i:02}',template=False) for i,(q,a) in enumerate(pairs)]
    # Counterfactual test pairs. Same final query; swap the fact or the final correction.
    # These forms are NOT used in training. Token vocab/subjects overlap deliberately.
    for i in range(4):
        for j,name in enumerate([names[i],names[i+4]]):
            challenge.append(row([f'Для этой проверки моё имя: {name}.','Принято.',
             'Какое имя принадлежит мне в этой беседе? Напиши только имя.',name+'.'],f'v21_pair_name_{i}_{j}','test'))
    for i in range(4):
        for j,col in enumerate([colors[i],colors[(i+2)%6]]):
            challenge.append(row([f'Для примера выбираю {["серую","розовую","фиолетовую","оранжевую"][i]} папку.','Понял.',
             f'Но итоговый выбор другой: {col} папка.','Учту.',
             'Каков итоговый цвет папки? Только цвет.',col+'.'],f'v21_pair_color_{i}_{j}','test'))
    for i in range(4):
        for j,place in enumerate([places[i],places[(i+3)%6]]):
            challenge.append(row([f'Сначала билет находился {["в кармане","в конверте","возле двери","в коробке"][i]}.','Понятно.',
             f'После перекладывания он {place}.','Принято.',
             'Назови текущее место билета без лишних слов.',place.capitalize()+'.'],f'v21_pair_place_{i}_{j}','test'))
    for i,(noun,yes,no) in enumerate([('письмо','Письмо уже отправил.','Письмо пока не отправлял.'),('билет','Билет уже куплен.','Билет пока не куплен.'),('рисунок','Рисунок закончен.','Рисунок пока не закончен.'),('встреча','Встреча подтверждена.','Встреча пока не подтверждена.')]):
        for j,fact in enumerate([yes,no]):
            question=['С отправкой письма уже закончили? Ответь да или нет.','Покупка билета уже состоялась? Ответь да или нет.','Работа над рисунком уже завершена? Ответь да или нет.','Договорённость о встрече уже окончательная? Ответь да или нет.'][i]
            challenge.append(row(['Сообщу точный статус. '+fact,'Понял.',question,'Да.' if j==0 else 'Нет.'],f'v21_pair_action_{i}_{j}','test'))
    return {'context':train,'natural':natural,'challenge':challenge}
if __name__=='__main__':
    print({k:len(v)for k,v in build().items()})
