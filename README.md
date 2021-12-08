# defeasible_conflict_gen

Model trained with code from [here](https://github.com/id4thomas/defeasible-nli)

## Inference Serving Overview
![nginx_upstream_server](./imgs/nginx_upstream.png)

```
└── nginx_configs - nginx Configuration Files
    └── gen_atomic.conf

└── inf_configs - HTTP Request configurations
    └── clf_config.json
    └── gen_config.json

# Server scripts
└── run_nginx.sh
└── run_streamlit.sh
└── run_flask_servers.sh

# Streamlit Page Files
└── streamlit_app.py - Main streamlit Page
└── st_gen_page.py - defines Generative Inference Page
└── st_clf_page.py - defines Classification Inference Page

# Flask Server Files
└── inference_gen.py - defines /predict_gen route
└── inference_clf.py - defines /predict_clf route

# Huggingface Transformer Model Inference Codes
└── defeasible_gen_model.py
└── defeasible_clf_model.py
└── utils.py
```

## Demo
### Generative Inference
![gen_atomic_example](./imgs/generative_atomic_sample.png)