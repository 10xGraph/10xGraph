# Getting started with Lumen

Lumen is a hosted dashboard service that turns event streams into shareable reports.

## Create a workspace

Sign in at app.lumen.example and choose **New workspace**. A workspace holds your data
sources, dashboards and teammates. Workspace names must be unique within your organisation.

## Connect a data source

Open **Sources** and pick Postgres, BigQuery or the Lumen HTTP collector. The HTTP collector
accepts JSON events at `https://collect.lumen.example/v1/events` with an `x-lumen-key` header.
The first events appear in the live view within about 10 seconds.

## Build your first dashboard

Press **New dashboard**, select a source, and drag a metric onto the canvas. Dashboards save
automatically. Use **Share** to create a read-only link.
