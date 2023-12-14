# segang-rl-assignment


base_train.py : TD3 기본 모델의 학습 프로그램 입니다.

base_reward_train.py : 보상 정책의 수정 후 TD3 기본 모델로 학습 합니다.

init_orth_train.py : base_reward_train 기반으로 각 레이어에 특성에 맞는 레이어 초기화 후 학습합니다.

init_orth_with_bais_train.py : init_orth_train 기반으로 bais 값을 0으로 초기화 후 학습합니다.


main.py : 가장 성능이 높게 나온 init_orth_with_bais_train 모델을 render_mode = 'human' 으로 실행합니다.

이때, 사전 학습된 모델의 가중치 값이 저장된 해당 파일(a66043_actor.pth, a66043_critic_1.pth, a66043_critic_2.pth) 과 init_orth_with_bais_train.py 가 같은 디렉토리 내에 존재하면 
오류 없이 실행됩니다.
