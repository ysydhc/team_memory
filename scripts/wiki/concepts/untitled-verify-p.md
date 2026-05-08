---
title: 
tags: []
source_ids: [verify-p1]
project: team_doc
experience_type: general
created_at: 
updated_at: 
confidence: medium
recall_count: 0
---

## 摘要

排查发现Docker容器内访问host.docker.internal超时。原因是docker-compose.yml里用了network_mode:bridge但没有extra_hosts映射。解决方案：在docker-compose.yml的postgres服务下添加extra_hosts:["host.docker.internal:host-gateway"]，重启容器后连接正常。

## 问题描述

排查发现Docker容器内访问host.docker.internal超时。原因是docker-compose.yml里用了network_mode:bridge但没有extra_hosts映射。解决方案：在docker-compose.yml的postgres服务下添加extra_hosts:["host.docker.internal:host-gateway"]，重启容器后连接正常。

## 解决方案

原因是docker-compose
解决方案：在docker-compose


## 来源

- experience_id: verify-p1
