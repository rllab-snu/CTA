# CTA
python main.py --env_name=scene-play-v0 --agent=agents/cta.py --agent.subgoal_steps=10 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=cube-single-play-v0 --agent=agents/cta.py --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=cube-double-play-v0 --agent=agents/cta.py --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=512
python main.py --env_name=cube-triple-play-v0 --agent=agents/cta.py --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=1024
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/cta.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/cta.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/cta.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=512
python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/cta.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=1024

# HIQL^\vee
python main.py --env_name=scene-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=10 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=cube-single-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=cube-double-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=512
python main.py --env_name=cube-triple-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=1024
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=512
python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/dualhiql.py --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=1024

# HIQL^\vee_{+\alpha^\vee}
python main.py --env_name=scene-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=10 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=cube-single-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=cube-double-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=512
python main.py --env_name=cube-triple-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=30 --agent.tr_latent_dim=8 --agent.batch_size=1024
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=256
python main.py --env_name=puzzle-4x5-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=512
python main.py --env_name=puzzle-4x6-play-v0 --agent=agents/dualhiql.py --agent.use_analogy=True --agent.subgoal_steps=20 --agent.tr_latent_dim=8 --agent.batch_size=1024